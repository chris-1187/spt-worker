import argparse
import os
import json
import time
import torch
import warnings
import numpy as np
from abc import ABC, abstractmethod
from typing import Iterator
from torch.utils.data import DataLoader
from tqdm import tqdm

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

    @abstractmethod
    def update(self, indices: torch.Tensor, chunk_logits: torch.Tensor, chunk_coords: torch.Tensor = None):
        # updates the global state with predictions from a single chunk
        pass

    @abstractmethod
    def get_final_predictions(self) -> torch.Tensor:
        # computes the final fusion and returns the class predictions
        pass


### Sampling strategies

class SlidingBlockSampler(PointChunkSampler):
    """
    Slides a spatial block over the point cloud (ignoring z-axis).
    Overlap is defined in meters.
    """

    def __init__(self, chunk_size: int, block_size_m: float = 8.0, overlap_m: float = 1.0, **kwargs):
        super().__init__(chunk_size, **kwargs)
        self.block_size_m = block_size_m
        self.overlap_m = overlap_m
        self.stride_m = block_size_m - overlap_m

        if self.stride_m <= 0:
            raise ValueError("Overlap must be less than block_size_m.")

    def generate_chunks(self, full_coord: torch.Tensor) -> Iterator[torch.Tensor]:
        min_xyz = full_coord.min(dim=0)[0]
        max_xyz = full_coord.max(dim=0)[0]

        # Slide across X and Y axis
        x_curr = min_xyz[0].item()

        while x_curr < max_xyz[0].item():
            y_curr = min_xyz[1].item()

            while y_curr < max_xyz[1].item():
                # Define spatial block size block
                x_min, x_max = x_curr, x_curr + self.block_size_m
                y_min, y_max = y_curr, y_curr + self.block_size_m

                # Boolean mask for points inside the bounding box
                mask = (full_coord[:, 0] >= x_min) & (full_coord[:, 0] < x_max) & \
                       (full_coord[:, 1] >= y_min) & (full_coord[:, 1] < y_max)

                indices = torch.nonzero(mask).squeeze(1)

                if len(indices) > 0:
                    # Limit chunk size
                    if len(indices) > self.chunk_size:
                        # Shuffle indices to ensure uniform sub-sampling across the block
                        indices = indices[torch.randperm(len(indices))]
                        for i in range(0, len(indices), self.chunk_size):
                            yield indices[i: i + self.chunk_size]
                    else:
                        yield indices

                y_curr += self.stride_m
            x_curr += self.stride_m


class HilbertSampler(PointChunkSampler):
    """
    1D Sliding Window Strategy: Sorts the raw 3D points into a 1D sequence
    using a hilbert curve, then slides a window across this sequence.
    Overlap is defined in points.
    """

    def __init__(self, chunk_size: int, overlap_points: int = 2000, grid_size: float = 0.01, **kwargs):
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
    Farthest Point Sampling + kNN Strategy.
    Finds uniform anchor points across the point cloud, then extracts the k nearest
    neighbors around each anchor.
    Overlap is defined indirectly by an oversampling factor.
    """

    def __init__(self, chunk_size: int, oversample_factor: float = 1.5, **kwargs):
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
    Voxel-guided + kNN Strategy.
    Finds uniform anchor points through voxelization. Uses sparse voxelization to get occupied regions.
    Uses the point closest to each occupied voxel's center as the anchor. Then extracts k nearest neighbors around each anchor.
    Overlap is defined indirectly by the voxel size.
    """

    def __init__(self, chunk_size: int, voxel_size: float = 6.0, **kwargs):
        super().__init__(chunk_size, **kwargs)
        self.voxel_size = voxel_size

    def generate_chunks(self, full_coord: torch.Tensor) -> Iterator[torch.Tensor]:
        total_points = full_coord.shape[0]

        if total_points <= self.chunk_size:
            yield torch.arange(total_points, device=full_coord.device)
            return

        # Sparse Voxelization (O(N)), shift coords to positive space for stable floor division
        min_coord = full_coord.min(dim=0)[0]
        shifted_coord = full_coord - min_coord

        # quantize points to voxel indices and extract the unique occupied voxels
        voxel_indices = torch.floor(shifted_coord / self.voxel_size).int()
        unique_voxels = torch.unique(voxel_indices, dim=0)

        # get voxel centers
        voxel_centers = (unique_voxels.float() + 0.5) * self.voxel_size + min_coord

        # get closest points for each voxel center by computing the distance from all voxel centers to all points
        dists_to_centers = torch.cdist(voxel_centers, full_coord)

        # argmin gives the index of the real point closest to each geometric center
        anchor_indices = torch.argmin(dists_to_centers, dim=1)
        anchors = full_coord[anchor_indices]  # [V, 3]

        # get k nearest neighbors around each anchor
        dists_to_anchors = torch.cdist(anchors, full_coord)
        _, knn_indices = torch.topk(dists_to_anchors, k=self.chunk_size, largest=False, dim=1)

        for i in range(anchors.shape[0]):
            yield knn_indices[i]


### Fusion strategies

class AverageLogitMerger(PredictionMerger):
    """
    Standard Average Logit Pooling over overlapping predictions.
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


### Strategy getter

def get_sampler(strategy: str, chunk_size: int, **kwargs) -> PointChunkSampler:
    strategy = strategy.lower()
    if strategy == 'block':
        block_size_m = kwargs.get('block_size_m', 8.0)
        overlap_m = kwargs.get('overlap_m', 1.5)
        return SlidingBlockSampler(chunk_size, block_size_m, overlap_m)
    elif strategy == 'hilbert':
        overlap_points = kwargs.get('overlap_size', 2000)
        return HilbertSampler(chunk_size, overlap_points)
    elif strategy == 'fps_knn':
        oversample_factor = kwargs.get('oversample_factor', 1.5)
        return FPSKNNSampler(chunk_size, oversample_factor)
    elif strategy == 'voxel_knn':
        voxel_size = kwargs.get('voxel_size', 6.0)
        return VoxelKNNSampler(chunk_size, voxel_size)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

def get_merger(strategy: str, total_points: int, num_classes: int, device: torch.device) -> PredictionMerger:
    strategy = strategy.lower()
    if strategy == 'logit_average':
        return AverageLogitMerger(total_points, num_classes, device)
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

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def per_class_acc(hist):
    return np.diag(hist) / hist.sum(1)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"> [Inference] Running on device: {device}")

    inference_dir = os.path.join(args.output_dir, "inference", args.inference_id)
    os.makedirs(inference_dir, exist_ok=True)

    chunk_size = args.chunk_size
    stride = chunk_size - args.overlap_size

    print(f"> [Inference] Loading checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    # Extract config if available
    if 'config' in checkpoint:
        print("> [Inference] Found configuration in checkpoint. Loading dynamic architecture...")
        model_config = checkpoint['config']['model_architecture']
    else:
        print("! [WARNING] No config found in checkpoint. Using HARDCODED defaults.")
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
        start_idx = args.start_idx,
        end_idx = args.end_idx
    )

    dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn
    )

    hist = np.zeros((num_classes, num_classes))
    inference_times = []

    print("> [System] Warming up JIT kernel. Compile Simt algos with nvrtc...")

    warmup_loader = DataLoader(
        val_dataset, batch_size=1, num_workers=args.num_workers,
        collate_fn=collate_fn, shuffle=False
    )
    warmup_batch = next(iter(warmup_loader))

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

    with torch.no_grad():
        for i, data_dict in tqdm(enumerate(dataloader), total=len(dataloader)):
            full_coord = data_dict['coord'].to(device)
            full_feat = data_dict['feat'].to(device)
            full_label = data_dict['label'].to(device)

            total_points = full_coord.shape[0]

            # initialize sampler
            sampler = get_sampler(
                args.sampling_strategy,
                chunk_size,
                overlap_size=args.overlap_size
            )

            # initialize fusion
            fusion = get_merger("logit_average", total_points, num_classes, device)

            torch.cuda.synchronize()
            t_start = time.time()

            for chunk_idx in sampler.generate_chunks(full_coord):
                chunk_coord = full_coord[chunk_idx]
                chunk_feat = full_feat[chunk_idx]
                chunk_batch_idx = torch.zeros(chunk_coord.shape[0], dtype=torch.long, device=device)

                chunk_input = {
                    "coord": chunk_coord,
                    "feat": chunk_feat,
                    "batch": chunk_batch_idx,
                    "grid_size": 0.01
                }

                # Forward pass
                output = model(chunk_input)
                logits = seg_head(output.feat)

                # Accumulate
                fusion.update(chunk_idx, logits, chunk_coord)

            torch.cuda.synchronize()
            inference_times.append(time.time() - t_start)

            # Fuse and evaluate
            final_preds = fusion.get_final_predictions()
            preds_np = final_preds.cpu().numpy()
            labels_np = full_label.cpu().numpy()

            valid_mask = labels_np != -1
            hist += fast_hist(preds_np[valid_mask], labels_np[valid_mask], num_classes)

            preds_kitti = kitti_map_array[preds_np]

            original_path = data_dict['file_path'][0]
            seq_id = original_path.split('/')[-3]
            frame_name = original_path.split('/')[-1].replace('.bin', '.label')
            save_dir = os.path.join(inference_dir, "predictions", seq_id)
            os.makedirs(save_dir, exist_ok=True)

            preds_kitti.tofile(os.path.join(save_dir, frame_name))

        # Construct JSON payload with raw counts
        node_metrics = {
            "node_name": args.node_name,
            "parameters": {
                "chunk_size": chunk_size,
                "overlap_size": args.overlap_size,
                "sampling_strategy": args.sampling_strategy,
                "sequences": args.sequences,
                "start_idx": args.start_idx,
                "end_idx": args.end_idx
            },
            "raw_metrics": {
                "hist": hist.tolist(),  # raw confusion matrix
                "total_inference_time_sec": sum(inference_times),
                "total_frames": len(inference_times)
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
    parser.add_argument('--start_idx', type=int, default=None, help="Start frame index for this worker")
    parser.add_argument('--end_idx', type=int, default=None, help="End frame index for this worker")
    parser.add_argument('--node_name', type=str, required=True, help="Name of the worker node")
    args = parser.parse_args()
    main(args)