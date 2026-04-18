from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .serialization import hilbert as hilbert_curve

IGNORE_INDEX = -1
LABEL_MAP = {
    0: IGNORE_INDEX,  # "unlabeled"
    1: IGNORE_INDEX,  # "outlier"
    10: 0,  # "car"
    11: 1,  # "bicycle"
    13: 4,  # "bus" -> "other-vehicle"
    15: 2,  # "motorcycle"
    16: 4,  # "on-rails" -> "other-vehicle"
    18: 3,  # "truck"
    20: 4,  # "other-vehicle"
    30: 5,  # "person"
    31: 6,  # "bicyclist"
    32: 7,  # "motorcyclist"
    40: 8,  # "road"
    44: 9,  # "parking"
    48: 10, # "sidewalk"
    49: 11, # "other-ground"
    50: 12, # "building"
    51: 13, # "fence"
    52: IGNORE_INDEX, # "other-structure" -> "unlabeled"
    60: 8,  # "lane-marking" -> "road"
    70: 14, # "vegetation"
    71: 15, # "trunk"
    72: 16, # "terrain"
    80: 17, # "pole"
    81: 18, # "traffic-sign"
    99: IGNORE_INDEX, # "other-object" -> "unlabeled"
    252: 0,  # "moving-car" -> "car"
    253: 6,  # "moving-bicyclist" -> "bicyclist"
    254: 5,  # "moving-person" -> "person"
    255: 7,  # "moving-motorcyclist" -> "motorcyclist"
    256: 4,  # "moving-on-rails" -> "other-vehicle"
    257: 4,  # "moving-bus" -> "other-vehicle"
    258: 3,  # "moving-truck" -> "truck"
    259: 4,  # "moving-other"-vehicle -> "other-vehicle"
}

class KittiSemanticDataset(Dataset):
    """
    PyTorch Dataset for the KITTI point cloud data.
    """

    def __init__(self,
                 root_dir: str,
                 labels_dir: str = None,
                 sequences: list[str] = None,
                 training: bool = None,
                 max_points: int = 10240,
                 sampling_strategy: str = 'hilbert'):
        """
        Args:
            root_dir (str): The root directory of the dataset, e.g., '.../kitti/dataset'.
            labels_dir (str, optional): The directory containing label files.
            sequences (list[str], optional): A list of sequence IDs to load (e.g., ['00', '02', '08']).
                                            If None, all sequences are loaded.
        """
        self.root_dir = Path(root_dir)
        self.labels_dir = Path(labels_dir) if labels_dir else None
        self.training = training
        self.sampling_strategy = sampling_strategy.lower()

        if self.sampling_strategy not in ['block', 'hilbert', 'fps_knn', 'voxel_knn', 'nuc_knn', 'kdtree_knn']:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}. Use 'block', 'hilbert', 'fps_knn', 'nuc_knn', 'kdtree_knn' or 'voxel_knn'.")

        self.max_points = max_points
        self.point_files = []
        self.label_files = []

        if sequences is None:
            # If no sequences are specified, load all
            print("> No sequences specified, loading all .bin files...")
            self.point_files = sorted(list(self.root_dir.glob("sequences/**/*.bin")))
            if self.labels_dir:
                self.label_files = sorted(list(self.labels_dir.glob("sequences/**/*.label")))
        else:
            # Load only the specified sequences
            print(f"> Loading specified sequences: {sequences}")
            for seq_id in sequences:
                self.point_files.extend(list(self.root_dir.glob(f"sequences/{seq_id}/**/*.bin")))
                if self.labels_dir:
                    self.label_files.extend(list(self.labels_dir.glob(f"sequences/{seq_id}/**/*.label")))

            # Sort the final lists to ensure they match
            self.point_files.sort()
            if self.labels_dir:
                self.label_files.sort()

        if self.labels_dir:
            assert len(self.point_files) > 0, "No point cloud files found."
            assert len(self.point_files) == len(self.label_files), \
                f"Mismatch: Found {len(self.point_files)} point files but {len(self.label_files)} label files."

    def __len__(self):
        return len(self.point_files)

    def _get_hilbert_indices(self, points, grid_size=0.01):
        """
        Sorts points based on Hilbert Curve index.
        O(NlogN) selection using numpy.argsort.
        Returns: sorted_indices (numpy array)
        """
        min_coord = points.min(axis=0)
        quantized = ((points - min_coord) / grid_size).astype(int)
        locs = torch.from_numpy(quantized).long()
        hilbert_codes = hilbert_curve.encode(locs, num_dims=3, num_bits=16)
        return torch.argsort(hilbert_codes).numpy()

    def _get_knn_indices(self, points, k):
        """
        Selects k points closest to a random center point.
        O(N) selection using numpy.argpartition.
        Returns: sorted_indices (numpy array)
        """
        num_points = points.shape[0]

        if num_points <= k:
            return np.arange(num_points)

        # Random center point index
        center_idx = np.random.randint(num_points)
        center_point = points[center_idx, :]

        # Compute Euclidean distances (Vectorized)
        # ||p - c||^2 is sufficient for ranking, avoids sqrt for speed
        dists = np.sum((points - center_point) ** 2, axis=1)

        # Find indices of the k nearest neighbors
        idx = np.argpartition(dists, k)[:k]

        return idx

    def _get_block_indices(self, points, block_size_m=8.0, min_points=1024):
        """
        Selects points within a random square spatial block.
        """
        num_points = points.shape[0]
        idx = np.array([])

        for _ in range(10):
            center_idx = np.random.randint(num_points)
            center_point = points[center_idx, :]

            min_bound = center_point - (block_size_m / 2)
            max_bound = center_point + (block_size_m / 2)

            mask = (points[:, 0] >= min_bound[0]) & (points[:, 0] < max_bound[0]) & \
                   (points[:, 1] >= min_bound[1]) & (points[:, 1] < max_bound[1])

            idx = np.nonzero(mask)[0]

            # break if dense enough
            if len(idx) >= min_points:
                break

        if len(idx) == 0:
            idx = np.array([center_idx])

        # downsample if needed
        if len(idx) > self.max_points:
            np.random.shuffle(idx)
            idx = idx[:self.max_points]

        return idx


    def __getitem__(self, idx):
        """
        Loads a single point cloud frame and its labels
        Returns a dictionary compatible with the Point Transformer V3 model
        """
        point_file_path = self.point_files[idx]
        points = np.fromfile(point_file_path, dtype=np.float32).reshape(-1, 4)
        # global ids for traceability
        original_indices = np.arange(len(points), dtype=np.int64)

        labels = None
        if self.labels_dir:
            label_file_path = self.label_files[idx]
            labels = np.fromfile(label_file_path, dtype=np.uint32).reshape(-1)
            semantic_labels_raw = (labels & 0xFFFF).astype(np.int32)
            semantic_labels = np.full_like(semantic_labels_raw, IGNORE_INDEX, dtype=np.int64)
            for raw_label, mapped_label in LABEL_MAP.items():
                semantic_labels[semantic_labels_raw == raw_label] = mapped_label

        if self.training:
            if self.sampling_strategy == 'block':
                block_idx = self._get_block_indices(points[:, :3], block_size_m=8.0)
                points = points[block_idx]
                original_indices = original_indices[block_idx]
                if labels is not None:
                    semantic_labels = semantic_labels[block_idx]

            elif self.sampling_strategy == 'hilbert':
                # 1. Sort by Hilbert
                sort_idx = self._get_hilbert_indices(points[:, :3])
                points = points[sort_idx]
                original_indices = original_indices[sort_idx]
                if labels is not None:
                    semantic_labels = semantic_labels[sort_idx]

                # 2. Slice Contiguous Chunk
                total_points = points.shape[0]
                if total_points > self.max_points:
                    valid_range_end = total_points - self.max_points
                    start_idx = np.random.randint(0, valid_range_end + 1)
                    end_idx = start_idx + self.max_points

                    points = points[start_idx:end_idx]
                    original_indices = original_indices[start_idx:end_idx]
                    if labels is not None:
                        semantic_labels = semantic_labels[start_idx:end_idx]

            elif self.sampling_strategy == 'knn':
                knn_idx = self._get_knn_indices(points[:, :3], k=self.max_points)
                points = points[knn_idx]
                original_indices = original_indices[knn_idx]
                if labels is not None:
                    semantic_labels = semantic_labels[knn_idx]


        # Prepare Output Dict
        data_dict = {
            "coord": torch.from_numpy(points[:, :3]),
            "feat": torch.from_numpy(points),  # XYZ + Intensity
            "index": torch.from_numpy(original_indices),  # for fusion
            "file_path": str(point_file_path)
        }

        if self.labels_dir:
            data_dict["label"] = torch.from_numpy(semantic_labels)

        return data_dict
