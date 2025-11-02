from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class KittiSemanticDataset(Dataset):
    """
    PyTorch Dataset for the KITTI point cloud data.
    """

    def __init__(self, root_dir: str, labels_dir: str = None, sequences: list[str] = None, max_points: int = 50000):
        """
        Args:
            root_dir (str): The root directory of the dataset, e.g., '.../kitti/dataset'.
            labels_dir (str, optional): The directory containing label files.
            sequences (list[str], optional): A list of sequence IDs to load (e.g., ['00', '02', '08']).
                                            If None, all sequences are loaded.
        """
        self.root_dir = Path(root_dir)
        self.labels_dir = Path(labels_dir) if labels_dir else None
        self.max_points = max_points

        self.point_files = []
        self.label_files = []

        if sequences is None:
            # If no sequences are specified, load all
            print("No sequences specified, loading all .bin files...")
            self.point_files = sorted(list(self.root_dir.glob("sequences/**/*.bin")))
            if self.labels_dir:
                self.label_files = sorted(list(self.labels_dir.glob("sequences/**/*.label")))
        else:
            # Load only the specified sequences
            print(f"Loading specified sequences: {sequences}")
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

    def __getitem__(self, idx):
        """
        Loads a single point cloud frame and its labels
        Returns a dictionary compatible with the Point Transformer V3 model
        """
        point_file_path = self.point_files[idx]
        points = np.fromfile(point_file_path, dtype=np.float32).reshape(-1, 4)

        labels = None
        if self.labels_dir:
            label_file_path = self.label_files[idx]
            labels = np.fromfile(label_file_path, dtype=np.uint32).reshape(-1)
            semantic_labels = (labels & 0xFFFF).astype(np.int64)

        if self.max_points is not None and points.shape[0] > self.max_points:
            # Generate random indices to select
            indices = np.random.choice(points.shape[0], self.max_points, replace=False)

            # Sample from points and labels
            points = points[indices]
            if labels is not None:
                semantic_labels = semantic_labels[indices]

        coords = torch.from_numpy(points[:, :3])  # XYZ
        feats = torch.from_numpy(points)  # XYZ + Intensity

        data_dict = {
            "coord": coords,
            "feat": feats,
        }

        if self.labels_dir:
            data_dict["label"] = torch.from_numpy(semantic_labels)

        return data_dict