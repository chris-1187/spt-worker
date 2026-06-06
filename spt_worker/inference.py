import argparse
import os
import json
import time
import torch
import warnings
import numpy as np
import redis
from abc import ABC, abstractmethod
from typing import Iterator
from torch.utils.data import DataLoader

from spt_worker.dataset import KittiSemanticDataset
from spt_worker.model import PointTransformerV3
from .serialization import hilbert as hilbert_curve

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(divide='ignore', invalid='ignore')


### Base classes

class PointChunkSampler(ABC):
    """
    Base class for all spatial chunking strategies.
    """
    def __init__(self, chunk_size: int, **kwargs):
        self.chunk_size = chunk_size

    @abstractmethod
    def generate_chunks(self, full_coord: torch.Tensor) -> Iterator[torch.Tensor]:
        pass


class PredictionMerger(ABC):
    """
    Base class to update and fuse final chunk predictions.
    """
    def __init__(self, total_points: int, num_classes: int, device: torch.device):
        self.total_points = total_points
        self.num_classes = num_classes
        self.device = device
        self.prediction_counts = torch.zeros(total_points, dtype=torch.long, device=device)

    @abstractmethod
    def update(self, indices: torch.Tensor, chunk_logits: torch.Tensor, chunk_coords: torch.Tensor = None):
        # updates the global state with predictions from a single chunk
        pass

    def get_average_oversampling(self) -> float:
        # average number of times a point was predicted across the frame
        return self.prediction_counts.float().mean().item()

    @abstractmethod
    def get_final_predictions(self) -> torch.Tensor:
        # computes the final fusion and returns the class predictions
        pass


### Sampling strategies

class SlidingBlockSampler(PointChunkSampler):
    """
    Planar sliding window partitioning (Baseline): Slides a fixed-size spatial block over the point cloud (ignoring z-axis).
    If a block exceeds the chunk limit, points are randomly dropped to fit. Empty blocks are ignored.
    """

    def __init__(self, chunk_size: int, block_size_m: float = 8.0, overlap_m: float = 0.5, **kwargs):
        super().__init__(chunk_size, **kwargs)
        self.block_size_m = block_size_m
        self.overlap_m = overlap_m
        self.stride_m = block_size_m - overlap_m

        if self.stride_m <= 0:
            raise ValueError("Overlap must be less than block_size_m.")

    def generate_chunks(self, full_coord: torch.Tensor) -> Iterator[torch.Tensor]:
        min_xyz = full_coord.min(dim=0)[0]
        max_xyz = full_coord.max(dim=0)[0]

        # slide across X and Y axis
        x_curr = min_xyz[0].item()

        while x_curr < max_xyz[0].item():
            y_curr = min_xyz[1].item()

            while y_curr < max_xyz[1].item():
                # define spatial block bounding box
                x_min, x_max = x_curr, x_curr + self.block_size_m
                y_min, y_max = y_curr, y_curr + self.block_size_m

                # boolean mask for points inside the bounding box
                mask = (full_coord[:, 0] >= x_min) & (full_coord[:, 0] < x_max) & \
                       (full_coord[:, 1] >= y_min) & (full_coord[:, 1] < y_max)

                indices = torch.nonzero(mask).squeeze(1)

                # drop empty tiles
                if len(indices) > 0:
                    # downsample randomly to chunk_size
                    if len(indices) > self.chunk_size:
                        perm = torch.randperm(len(indices), device=indices.device)[:self.chunk_size]
                        indices = indices[perm]

                    yield indices

                y_curr += self.stride_m
            x_curr += self.stride_m


class HilbertSampler(PointChunkSampler):
    """
    Sequential patching via hilbert curve serialization: Sorts the raw 3D points into a 1D sequence using a hilbert
    curve, then slides a window across this sequence. Overlap is defined in points.
    """

    def __init__(self, chunk_size: int, overlap_points: int = 5000, grid_size: float = 0.01, **kwargs):
        super().__init__(chunk_size, **kwargs)
        self.overlap_points = overlap_points
        self.stride = chunk_size - overlap_points
        self.grid_size = grid_size

        if self.stride <= 0:
            raise ValueError("Overlap must be less than chunk_size.")

    def _get_hilbert_indices(self, points: torch.Tensor) -> torch.Tensor:
        min_coord = points.min(dim=0)[0]
        quantized = ((points - min_coord) / self.grid_size).long()
        hilbert_codes = hilbert_curve.encode(quantized, num_dims=3, num_bits=16)
        return torch.argsort(hilbert_codes)

    def generate_chunks(self, full_coord: torch.Tensor) -> Iterator[torch.Tensor]:
        total_points = full_coord.shape[0]
        sorted_global_indices = self._get_hilbert_indices(full_coord)

        # sliding window
        start = 0
        while start < total_points:
            end = min(start + self.chunk_size, total_points)

            # return the actual global indices
            yield sorted_global_indices[start:end]

            if end == total_points:
                break
            start += self.stride


class FPSKNNSampler(PointChunkSampler):
    """
    Uniform anchoring via farthest point sampling (FPS): Finds uniform anchor points across the point cloud, then
    extracts the k nearest neighbors around each anchor. Overlap is defined indirectly by an oversampling factor.
    """

    def __init__(self, chunk_size: int, oversample_factor: float = 6.0, **kwargs):
        super().__init__(chunk_size, **kwargs)
        self.oversample_factor = oversample_factor

    def _fps(self, coords: torch.Tensor, num_samples: int) -> torch.Tensor:
        N = coords.shape[0]
        centroids = torch.zeros(num_samples, dtype=torch.long, device=coords.device)
        distance = torch.ones(N, device=coords.device) * 1e10

        # Pick the first point randomly
        farthest = torch.randint(0, N, (1,), dtype=torch.long, device=coords.device)

        for i in range(num_samples):
            centroids[i] = farthest
            centroid = coords[farthest, :].view(1, 3)
            # calculate squared euclidean distance to current centroid
            dist = torch.sum((coords - centroid) ** 2, dim=-1)
            # update minimum distances
            mask = dist < distance
            distance[mask] = dist[mask]
            # next centroid is the one farthest from all existing centroids
            farthest = torch.max(distance, dim=-1)[1]

        return centroids

    def generate_chunks(self, full_coord: torch.Tensor) -> Iterator[torch.Tensor]:
        total_points = full_coord.shape[0]

        # If the frame is smaller than one chunk, just yield everything once
        if total_points <= self.chunk_size:
            yield torch.arange(total_points, device=full_coord.device)
            return

        # Calculate how many chunk centers we need to guarantee coverage based on frame size, chunk size, and oversampling factor
        base_chunks = np.ceil(total_points / self.chunk_size)
        num_anchors = int(base_chunks * self.oversample_factor)

        anchor_indices = self._fps(full_coord, num_anchors)
        anchors = full_coord[anchor_indices]  # [M, 3]

        # compute pairwise distances between anchors [M, 3] and full_coord [N, 3]
        dists = torch.cdist(anchors, full_coord)  # [M, N]

        # get indices of the K smallest distances for each anchor
        _, knn_indices = torch.topk(dists, k=self.chunk_size, largest=False, dim=1)

        for i in range(num_anchors):
            yield knn_indices[i]


class VoxelKNNSampler(PointChunkSampler):
    """
    Sparse voxel-guided anchoring: Finds uniform anchor points through voxelization. Uses sparse voxelization to get
    occupied regions. Uses the point closest to each occupied voxel's center as the anchor and extracts k nearest
    neighbors around each. Overlap is defined indirectly by the voxel size.
    """

    def __init__(self, chunk_size: int, voxel_size: float = 8.0, **kwargs):
        super().__init__(chunk_size, **kwargs)
        self.voxel_size = voxel_size

    def generate_chunks(self, full_coord: torch.Tensor) -> Iterator[torch.Tensor]:
        total_points = full_coord.shape[0]

        if total_points <= self.chunk_size:
            yield torch.arange(total_points, device=full_coord.device)
            return

        # sparse voxelization: shift coords to positive space for stable floor division
        min_coord = full_coord.min(dim=0)[0]
        shifted_coord = full_coord - min_coord

        # quantize points to voxel indices and extract the unique occupied voxels
        voxel_indices = torch.floor(shifted_coord / self.voxel_size).int()
        unique_voxels, counts = torch.unique(voxel_indices, dim=0, return_counts=True)
        min_points_threshold = 50
        valid_mask = counts >= min_points_threshold
        unique_voxels = unique_voxels[valid_mask]

        # failsafe
        if len(unique_voxels) == 0:
            unique_voxels = torch.unique(voxel_indices, dim=0)

        # get voxel centers
        voxel_centers = (unique_voxels.float() + 0.5) * self.voxel_size + min_coord

        # get closest points for each voxel center
        dists_to_centers = torch.cdist(voxel_centers, full_coord)

        # argmin gives the index of the real point closest to each geometric center
        anchor_indices = torch.argmin(dists_to_centers, dim=1)
        anchors = full_coord[anchor_indices]  # [V, 3]

        # get k nearest neighbors around each anchor
        dists_to_anchors = torch.cdist(anchors, full_coord)
        _, knn_indices = torch.topk(dists_to_anchors, k=self.chunk_size, largest=False, dim=1)

        for i in range(anchors.shape[0]):
            yield knn_indices[i]

class KDTreeKNNSampler(PointChunkSampler):
    """
    Recursive kd-tree partitioning: Divides the point cloud along its widest axis at the median point until every
    spatial leaf contains fewer than `leaf_size` points. Then extracts the geometric centroid of each leaf as kNN anchor.
    """
    def __init__(self, chunk_size: int, leaf_size: int = 4000, **kwargs): # 4000 = 32 chunks, 8000 = 16 chunks
        super().__init__(chunk_size, **kwargs)
        self.leaf_size = leaf_size

    def _build_tree_and_get_centroids(self, coords: torch.Tensor) -> list:
        num_points = coords.shape[0]

        # leaf node has dropped below the target leaf_size threshold
        if num_points <= self.leaf_size:
            return [coords.mean(dim=0)]

        # find the dimension with the largest physical extent
        mins = coords.min(dim=0)[0]
        maxs = coords.max(dim=0)[0]
        extents = maxs - mins
        split_dim = torch.argmax(extents).item()

        # find median value along that dimension to ensure 50/50 point split
        median_val = torch.median(coords[:, split_dim])

        # boolean masks to split the points
        left_mask = coords[:, split_dim] < median_val
        right_mask = ~left_mask

        if not left_mask.any() or not right_mask.any():
            return [coords.mean(dim=0)]

        # recursive call into sub-boxes
        left_centroids = self._build_tree_and_get_centroids(coords[left_mask])
        right_centroids = self._build_tree_and_get_centroids(coords[right_mask])

        return left_centroids + right_centroids

    def generate_chunks(self, full_coord: torch.Tensor) -> Iterator[torch.Tensor]:
        total_points = full_coord.shape[0]
        if total_points <= self.chunk_size:
            yield torch.arange(total_points, device=full_coord.device)
            return

        # generate leaf centroids recursively
        centroids_list = self._build_tree_and_get_centroids(full_coord)
        voxel_centers = torch.stack(centroids_list)

        # get anchors
        dists_to_centers = torch.cdist(voxel_centers, full_coord)
        anchor_indices = torch.argmin(dists_to_centers, dim=1)
        anchors = full_coord[anchor_indices]

        # get kNN Spheres around the anchors
        dists_to_anchors = torch.cdist(anchors, full_coord)
        _, knn_indices = torch.topk(dists_to_anchors, k=self.chunk_size, largest=False, dim=1)

        for i in range(anchors.shape[0]):
            yield knn_indices[i]


class NUCVoxelKNNSampler(PointChunkSampler):
    """
    Non-uniform anchoring via cylindrical partitioning: Dynamically expands voxel boundaries radially using Arithmetic
    Progression of Interval (API) Strategy, to adjust to non-unform LiDAR point distribution.
    """

    def __init__(self, chunk_size: int, a0: float = 2.0, d: float = 1.0, angular_bins: int = 8, z_bins: int = 1,
                 **kwargs):
        super().__init__(chunk_size, **kwargs)
        self.a0 = a0  # initial voxel width at sensor
        self.d = d  # radial expansion steps
        self.angular_bins = angular_bins
        self.z_bins = z_bins

    def generate_chunks(self, full_coord: torch.Tensor) -> Iterator[torch.Tensor]:
        total_points = full_coord.shape[0]
        if total_points <= self.chunk_size:
            yield torch.arange(total_points, device=full_coord.device)
            return

        x = full_coord[:, 0]
        y = full_coord[:, 1]
        z = full_coord[:, 2]

        # transform to cylindrical coordinates (r, theta, z)
        r = torch.sqrt(x ** 2 + y ** 2)
        theta = torch.atan2(y, x)

        # API radial quantization
        discriminant = (self.a0 - 0.5 * self.d) ** 2 + 2 * self.d * r
        discriminant = torch.clamp(discriminant, min=0.0)
        r_idx = torch.floor((-(self.a0 - 0.5 * self.d) + torch.sqrt(discriminant)) / self.d).int()

        # uniform angular & z quantization
        theta_idx = torch.floor(((theta + np.pi) / (2 * np.pi)) * self.angular_bins).int()

        z_min = z.min()
        z_max = z.max()
        z_range = torch.clamp(z_max - z_min, min=1e-3)
        z_idx = torch.floor(((z - z_min) / z_range) * self.z_bins).int()
        z_idx = torch.clamp(z_idx, max=self.z_bins - 1)  # safeguard upper boundary

        # extract only unique and occupied voxels
        voxel_indices = torch.stack([r_idx, theta_idx, z_idx], dim=1)
        unique_voxels, counts = torch.unique(voxel_indices, dim=0, return_counts=True)
        min_points_threshold = 50
        valid_mask = counts >= min_points_threshold
        unique_voxels = unique_voxels[valid_mask]

        if len(unique_voxels) == 0:
            unique_voxels = torch.unique(voxel_indices, dim=0)

        # print(f"DEBUG: Generated {unique_voxels.shape[0]} NUC-API anchors.") # validation

        # convert voxel centers back to cartesian space
        u_r_idx = unique_voxels[:, 0].float()
        u_theta_idx = unique_voxels[:, 1].float()
        u_z_idx = unique_voxels[:, 2].float()

        # get geometric centers
        r_min_bound = u_r_idx * self.a0 + 0.5 * self.d * u_r_idx * (u_r_idx - 1)
        r_max_bound = (u_r_idx + 1) * self.a0 + 0.5 * self.d * (u_r_idx + 1) * u_r_idx
        r_center = (r_min_bound + r_max_bound) / 2.0

        theta_center = (u_theta_idx + 0.5) * (2 * np.pi / self.angular_bins) - np.pi
        z_center = (u_z_idx + 0.5) * (z_range / self.z_bins) + z_min

        center_x = r_center * torch.cos(theta_center)
        center_y = r_center * torch.sin(theta_center)
        voxel_centers = torch.stack([center_x, center_y, z_center], dim=1)  # [V, 3]

        # get anchors
        dists_to_centers = torch.cdist(voxel_centers, full_coord)
        anchor_indices = torch.argmin(dists_to_centers, dim=1)
        anchors = full_coord[anchor_indices]

        # get kNN Spheres around the anchors
        dists_to_anchors = torch.cdist(anchors, full_coord)
        _, knn_indices = torch.topk(dists_to_anchors, k=self.chunk_size, largest=False, dim=1)

        for i in range(anchors.shape[0]):
            yield knn_indices[i]


### Fusion strategies

class AverageLogitMerger(PredictionMerger):
    """
    Average logit pooling (Baseline): Tracks and averages the raw logits across all tiles.
    """

    def __init__(self, total_points: int, num_classes: int, device: torch.device):
        super().__init__(total_points, num_classes, device)
        # Allocate global state tensors on the GPU
        self.full_logits = torch.zeros((total_points, num_classes), device=device)
        self.full_count = torch.zeros((total_points, 1), device=device)

    def update(self, indices: torch.Tensor, chunk_logits: torch.Tensor, chunk_coords: torch.Tensor = None):
        # Adds the raw logits to the global accumulator based on the provided indices
        self.full_logits[indices] += chunk_logits
        self.full_count[indices] += 1.0
        self.prediction_counts[indices] += 1


    def get_final_predictions(self) -> torch.Tensor:
        # prevent division by zero
        valid_mask = self.full_count.squeeze(1) > 0

        # averaging logits
        self.full_logits[valid_mask] /= self.full_count[valid_mask]

        # initialize output with 0 (which maps to IGNORE_INDEX in the kitti dataset)
        final_preds = torch.zeros(self.total_points, dtype=torch.long, device=self.device)

        # argmax along the class dimension
        final_preds[valid_mask] = self.full_logits[valid_mask].argmax(dim=1)

        return final_preds


class UncertaintyMerger(PredictionMerger):
    """
    Stochastic inference via Monte Carlo dropout: Keeps the prediction from the tile that exhibits the lowest
    predictive entropy.
    """
    def __init__(self, total_points: int, num_classes: int, device: torch.device):
        super().__init__(total_points, num_classes, device)
        self.best_entropy = torch.full((total_points,), float('inf'), device=device)
        self.best_probs = torch.zeros((total_points, num_classes), device=device)

    def update(self, indices: torch.Tensor, **kwargs):
        expected_probs = kwargs['expected_probs']
        entropy = kwargs['uncertainty']

        self.prediction_counts[indices] += 1

        # identify which points in the incoming chunk have better (lower) entropy
        current_best_entropy = self.best_entropy[indices]
        update_mask = entropy < current_best_entropy

        if update_mask.any():
            global_update_indices = indices[update_mask]
            self.best_entropy[global_update_indices] = entropy[update_mask]
            self.best_probs[global_update_indices] = expected_probs[update_mask]

    def get_final_predictions(self) -> torch.Tensor:
        return self.best_probs.argmax(dim=1)


### Strategy getter

def get_sampler(strategy: str, chunk_size: int, **kwargs) -> PointChunkSampler:
    strategy = strategy.lower()
    if strategy == 'block':
        block_size_m = kwargs.get('block_size_m', 8.0)
        overlap_m = kwargs.get('overlap_m', 0.5)
        return SlidingBlockSampler(chunk_size, block_size_m, overlap_m)
    elif strategy == 'hilbert':
        overlap_points = kwargs.get('overlap_size', 5000)
        return HilbertSampler(chunk_size, overlap_points)
    elif strategy == 'fps_knn':
        oversample_factor = kwargs.get('oversample_factor', 6.0)
        return FPSKNNSampler(chunk_size, oversample_factor)
    elif strategy == 'voxel_knn':
        voxel_size = kwargs.get('voxel_size', 8.0)
        return VoxelKNNSampler(chunk_size, voxel_size)
    elif strategy == 'kdtree_knn':
        leaf_size = kwargs.get('leaf_size', 4000)
        return KDTreeKNNSampler(chunk_size, leaf_size=leaf_size)
    elif strategy == 'nuc_knn':
        a0 = kwargs.get('a0', 2.0)
        d = kwargs.get('d', 1.0)
        angular_bins = kwargs.get('angular_bins', 8)
        z_bins = kwargs.get('z_bins', 1)
        return NUCVoxelKNNSampler(chunk_size, a0, d, angular_bins, z_bins)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

def get_merger(strategy: str, total_points: int, num_classes: int, device: torch.device) -> PredictionMerger:
    strategy = strategy.lower()
    if strategy == 'logit_average':
        return AverageLogitMerger(total_points, num_classes, device)
    elif strategy == 'mc_uncertainty':
        return UncertaintyMerger(total_points, num_classes, device)
    else:
        raise ValueError(f"Unknown fusion strategy: {strategy}")


### Helper functions

def collate_fn(batch):
    collated = {}
    batch_indices = []
    for i, sample in enumerate(batch):
        num_points = sample['coord'].shape[0]
        batch_indices.append(torch.full((num_points,), i, dtype=torch.long))
        for key, value in sample.items():
            if key not in collated:
                collated[key] = []
            collated[key].append(value)
    for key in collated:
        if isinstance(collated[key][0], torch.Tensor):
            collated[key] = torch.cat(collated[key], dim=0)
    collated['batch'] = torch.cat(batch_indices, dim=0)
    return collated

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"> [Inference] Running on device: {device}")

    inference_dir = os.path.join(args.output_dir, "inference", args.inference_id)
    os.makedirs(inference_dir, exist_ok=True)
    chunk_size = args.chunk_size

    # Redis connection
    print(f"> [Inference] Connecting to Redis Queue at {args.redis_host}...")
    try:
        redis_client = redis.Redis(host=args.redis_host, port=6379, db=0)
        redis_client.ping()
    except redis.ConnectionError:
        print("[ERROR] Could not connect to Redis server. Exiting.")
        return

    print(f"> [Inference] Loading checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    # Extract config if available
    if 'config' in checkpoint:
        print("> [Inference] Found configuration in checkpoint. Loading dynamic architecture...")
        model_config = checkpoint['config']['model_architecture']
    else:
        print("! [WARNING] No config found in checkpoint. Using hardcoded defaults.")
        # Fallback
        model_config = {
            "in_channels": 4,
            "enable_flash": False,
            "enc_channels": (32, 64, 128, 128, 256),
            "dec_channels": (32, 32, 64, 128),
            "enc_depths": (2, 2, 2, 2, 2),
            "dec_depths": (2, 2, 2, 2),
            "enc_num_head": (2, 4, 8, 8, 16),
            "dec_num_head": (4, 4, 8, 8),
            "enc_patch_size": (256, 256, 256, 256, 256),
            "dec_patch_size": (256, 256, 256, 256)
        }

    model = PointTransformerV3(**model_config).to(device)

    # Load Weights
    model.load_state_dict(checkpoint['model_state_dict'])

    num_classes = 19
    seg_head = torch.nn.Linear(32, num_classes).to(device)
    seg_head.load_state_dict(checkpoint['seg_head_state_dict'])

    model.eval()
    seg_head.eval()

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"> [Model] Total Parameters: {params / 1e6:.2f} M")

    # Dataset
    val_dataset = KittiSemanticDataset(
        root_dir=args.data_path,
        labels_dir=args.labels_path,
        sequences=args.sequences,
        training=False,
        max_points=None,
        sampling_strategy=args.sampling_strategy,
    )

    if len(val_dataset) == 0:
        print("[ERROR] No files found in dataset.")
        return

    hist = np.zeros((num_classes, num_classes))

    # Init metrics
    sampling_times_ms = []
    model_times_ms = []
    fusion_times_ms = []
    e2e_times_ms = []
    oversampling_factors = []
    tiles_per_frame = []
    total_cleanup_iterations = 0
    interpolated_percentages_list = []

    print("> [System] Warming up JIT kernel. Compile Simt algos with nvrtc...")

    warmup_batch = collate_fn([val_dataset[0]])
    warmup_batch['grid_size'] = 0.01
    for key, value in warmup_batch.items():
        if isinstance(value, torch.Tensor):
            warmup_batch[key] = value.to(device, non_blocking=True)

    with torch.no_grad():
        model(warmup_batch)

    print("> [System] Algos compiled. Starting inference evaluation...")

    kitti_map_array = np.array([
        10, 11, 15, 18, 20, 30, 31, 32,
        40, 44, 48, 49, 50, 51, 70,
        71, 72, 80, 81
    ], dtype=np.uint32)

    use_mc_dropout = args.fusion_strategy == 'mc_uncertainty'
    if use_mc_dropout:
        print(f"> [MC Dropout] Enabled with {args.mc_passes} stochastic forward passes per chunk.")
        # Set dropout layers into train() mode to enable stochastic sampling, while leaving BatchNorm/LayerNorm in eval() mode
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    use_interpolation_cleanup = args.sampling_strategy in ['block', 'hilbert', 'voxel_knn', 'nuc_knn', 'kdtree_knn', 'fps_knn']

    frames_processed = 0

    with torch.no_grad():
        while True:
            # atomic popping from redis queue
            raw_idx = redis_client.lpop(args.redis_queue)

            if raw_idx is None:
                # queue is empty -> break the loop
                print(f"\n> [Worker {args.node_name}] Queue empty. Spinning down.")
                break

            redis_client.expire(args.redis_queue, 86400)

            dataset_idx = int(raw_idx)

            if frames_processed % 10 == 0:
                print(f"  -> [Worker {args.node_name}] Fetched and processing frame index {dataset_idx}...")

            # fetch one frame and collate
            data_dict_raw = val_dataset[dataset_idx]
            data_dict = collate_fn([data_dict_raw])

            # aggregate node compute time (end to end)
            t_e2e_start = time.time()

            full_coord = data_dict['coord'].to(device)
            full_feat = data_dict['feat'].to(device)
            full_label = data_dict['label'].to(device)

            total_points = full_coord.shape[0]

            # initialize sampler
            sampler = get_sampler(
                args.sampling_strategy,
                chunk_size,
            )

            # initialize fusion
            fusion = get_merger(
                args.fusion_strategy,
                total_points,
                num_classes,
                device
            )

            torch.cuda.synchronize()
            t_sample_start = time.time() # measure time for sampling

            chunks = list(sampler.generate_chunks(full_coord))
            num_tiles = len(chunks)

            torch.cuda.synchronize()
            frame_sample_ms = (time.time() - t_sample_start) * 1000.0

            # async event collectors
            model_events = []
            fusion_events = []

            for chunk_idx in chunks:
                chunk_coord = full_coord[chunk_idx]
                chunk_feat = full_feat[chunk_idx]
                chunk_batch_idx = torch.zeros(chunk_coord.shape[0], dtype=torch.long, device=device)

                chunk_input = {
                    "coord": chunk_coord,
                    "feat": chunk_feat,
                    "batch": chunk_batch_idx,
                    "grid_size": 0.01
                }

                m_start, m_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                f_start, f_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

                if use_mc_dropout:
                    m_start.record()
                    chunk_probs_list = []
                    # run M stochastic passes
                    for _ in range(args.mc_passes):
                        output = model(chunk_input)
                        logits = seg_head(output.feat)
                        chunk_probs_list.append(torch.softmax(logits, dim=1))
                    m_end.record()
                    model_events.append((m_start, m_end))
                    f_start.record()
                    # per-tile prediction and uncertainty calculation
                    stacked_probs = torch.stack(chunk_probs_list, dim=0)  # [M, N, C]
                    expected_probs = stacked_probs.mean(dim=0)  # [N, C]

                    # Shannon Entropy: -sum(P * log(P))
                    eps = 1e-10
                    entropy = -torch.sum(expected_probs * torch.log(expected_probs + eps), dim=1)  # [N]

                    fusion.update(chunk_idx, expected_probs=expected_probs, uncertainty=entropy)

                    f_end.record()
                    fusion_events.append((f_start, f_end))

                else:
                    m_start.record()
                    # logit averaging baseline
                    output = model(chunk_input)
                    logits = seg_head(output.feat)
                    m_end.record()
                    model_events.append((m_start, m_end))
                    f_start.record()
                    fusion.update(chunk_idx, chunk_logits=logits)
                    f_end.record()
                    fusion_events.append((f_start, f_end))

            def get_missed_mask(fusion_merger):
                if isinstance(fusion_merger, AverageLogitMerger):
                    return fusion_merger.full_count.squeeze(1) == 0
                elif isinstance(fusion_merger, UncertaintyMerger):
                    return fusion_merger.best_entropy == float('inf')
                return torch.zeros(total_points, dtype=torch.bool, device=device)

            num_interpolated = 0

            torch.cuda.synchronize()
            frame_model_ms = sum(s.elapsed_time(e) for s, e in model_events)
            frame_fusion_ms = sum(s.elapsed_time(e) for s, e in fusion_events)

            # Fuse and evaluate
            t_final_fusion_start = time.time()
            final_preds = fusion.get_final_predictions()


            if use_interpolation_cleanup:
                unpredicted_mask = get_missed_mask(fusion)
                predicted_mask = ~unpredicted_mask
                num_interpolated = unpredicted_mask.sum().item()

                if num_interpolated > 0 and num_interpolated < total_points:
                    print(f"  -> [Warning] Fallback Nearest-Neighbor interpolating {num_interpolated} missed points.")

                    predicted_coords = full_coord[predicted_mask]
                    unpredicted_coords = full_coord[unpredicted_mask]

                    # global indices
                    unpred_global_indices = torch.nonzero(unpredicted_mask).squeeze(1)
                    pred_global_indices = torch.nonzero(predicted_mask).squeeze(1)

                    batch_size = 2000

                    for i in range(0, unpredicted_coords.shape[0], batch_size):
                        end_idx = min(i + batch_size, unpredicted_coords.shape[0])
                        sub_batch_unpred = unpredicted_coords[i:end_idx]

                        sub_dists = torch.cdist(sub_batch_unpred, predicted_coords)

                        # get local index of the closest predicted point
                        closest_local_indices = torch.argmin(sub_dists, dim=1)

                        # map local index back to global index
                        closest_global_indices = pred_global_indices[closest_local_indices]

                        nearest_labels = final_preds[closest_global_indices]

                        # assign copied label back to original output tensor
                        batch_global_unpred_indices = unpred_global_indices[i:end_idx]
                        final_preds[batch_global_unpred_indices] = nearest_labels

            interpolated_percentages_list.append((num_interpolated / total_points) * 100.0)

            preds_np = final_preds.cpu().numpy()
            labels_np = full_label.cpu().numpy()

            valid_mask = labels_np != -1
            hist += fast_hist(preds_np[valid_mask], labels_np[valid_mask], num_classes)

            preds_kitti = kitti_map_array[preds_np]
            frame_fusion_ms += (time.time() - t_final_fusion_start) * 1000.0

            original_path = data_dict['file_path'][0]
            seq_id = original_path.split('/')[-3]
            frame_name = original_path.split('/')[-1].replace('.bin', '.label')
            save_dir = os.path.join(inference_dir, "predictions", seq_id)
            os.makedirs(save_dir, exist_ok=True)

            preds_kitti.tofile(os.path.join(save_dir, frame_name))

            frame_e2e_ms = (time.time() - t_e2e_start) * 1000.0

            sampling_times_ms.append(frame_sample_ms)
            model_times_ms.append(frame_model_ms)
            fusion_times_ms.append(frame_fusion_ms)
            e2e_times_ms.append(frame_e2e_ms)
            oversampling_factors.append(fusion.get_average_oversampling())
            tiles_per_frame.append(num_tiles)

            frames_processed += 1

        # Construct JSON payload with raw counts
        node_metrics = {
            "node_name": args.node_name,
            "parameters": {
                "chunk_size": chunk_size,
                "overlap_size": args.overlap_size,
                "sampling_strategy": args.sampling_strategy,
                "fusion_strategy": args.fusion_strategy,
                "fusion_method": args.fusion_method,
                "mc_passes": args.mc_passes,
                "sequences": args.sequences,
                "redis_queue_name": args.redis_queue
            },
            "raw_metrics": {
                "total_sampling_time_sec": sum(sampling_times_ms) / 1000.0,
                "total_model_time_sec": sum(model_times_ms) / 1000.0,
                "total_fusion_time_sec": sum(fusion_times_ms) / 1000.0,
                "total_e2e_time_sec": sum(e2e_times_ms) / 1000.0,
                "total_frames": len(e2e_times_ms),
                "average_oversampling": sum(oversampling_factors) / len(oversampling_factors) if oversampling_factors else 1.0,
                "average_interpolated_percentage": sum(interpolated_percentages_list) / len(
                    interpolated_percentages_list) if interpolated_percentages_list else 0.0,
                "cleanup_iterations": total_cleanup_iterations,
                "total_tiles": sum(tiles_per_frame),
                "hist": hist.tolist()
            }
        }

        # Save as a node-specific partial file
        metrics_file = os.path.join(inference_dir, f"evaluation_metrics_{args.node_name}.json")
        with open(metrics_file, "w") as f:
            json.dump(node_metrics, f, indent=4)

        print(f"> [Inference] Worker {args.node_name} complete. Wrote partial metrics to {metrics_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--labels_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the root experiment directory")
    parser.add_argument('--inference_id', type=str, required=True)
    parser.add_argument('--sequences', type=str, nargs='+', default=['08'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--chunk_size', type=int, default=40000)
    parser.add_argument('--overlap_size', type=int, default=2000)
    parser.add_argument('--sampling_strategy', type=str, default='hilbert')
    parser.add_argument('--fusion_strategy', type=str, default='logit_average', choices=['logit_average', 'mc_uncertainty'])
    parser.add_argument('--mc_passes', type=int, default=10)
    parser.add_argument('--redis_host', type=str, required=True, help="IP of the primary node hosting the redis queue")
    parser.add_argument('--redis_queue', type=str, required=True, help="Key of the redis list")
    parser.add_argument('--node_name', type=str, required=True, help="Name of the worker node")
    args = parser.parse_args()
    main(args)