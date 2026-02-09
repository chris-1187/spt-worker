import argparse
import os
import torch
import numpy as np
import time
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from torch.utils.data import DataLoader
from tqdm import tqdm
from tabulate import tabulate

from spt_worker.dataset import KittiSemanticDataset
from spt_worker.model import PointTransformerV3


class SuppressOutput:
    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self._original_stdout = os.dup(1)
        self._original_stderr = os.dup(2)
        self._null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._null, 1)
        os.dup2(self._null, 2)

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self._original_stdout, 1)
        os.dup2(self._original_stderr, 2)
        os.close(self._null)
        os.close(self._original_stdout)
        os.close(self._original_stderr)

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
    print(f"> [Eval] Running on device: {device}")

    print(f"> [Eval] Loading checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    # Extract config if available
    if 'config' in checkpoint:
        print("> [Eval] Found configuration in checkpoint. Loading dynamic architecture...")
        model_config = checkpoint['config']['model_architecture']
    else:
        print("! [WARNING] No config found in checkpoint. Using HARDCODED defaults (Risk of mismatch!).")
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

    out_channels = 32
    num_classes = 19
    seg_head = torch.nn.Linear(out_channels, num_classes).to(device)
    seg_head.load_state_dict(checkpoint['seg_head_state_dict'])

    model.eval()
    seg_head.eval()

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"> [Model] Total Parameters: {params / 1e6:.2f} M")

    # Dataset
    val_sequences = args.sequences if args.sequences else ['08']
    val_dataset = KittiSemanticDataset(
        root_dir=args.data_path,
        labels_dir=args.labels_path,
        sequences=val_sequences,
        training=False,
        max_points=40000  # High res for validation!
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
        with SuppressOutput():
            model(warmup_batch)

    print("> [System] Algos compiled. Starting inference evaluation...")

    with torch.no_grad():
        for i, data_dict in tqdm(enumerate(dataloader), total=len(dataloader)):
            data_dict['grid_size'] = 0.01
            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor):
                    data_dict[key] = value.to(device, non_blocking=True)
                else:
                    data_dict[key] = value

            # Timing Inference
            torch.cuda.synchronize()
            start_time = time.time()

            output = model(data_dict)
            logits = seg_head(output.feat)
            preds = logits.argmax(dim=1)

            torch.cuda.synchronize()
            inference_times.append(time.time() - start_time)

            labels = data_dict['label']

            # Metrics
            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            valid_mask = labels_np != -1
            hist += fast_hist(preds_np[valid_mask], labels_np[valid_mask], num_classes)

    # Final Metrics
    ious = per_class_iu(hist) * 100
    accs = per_class_acc(hist) * 100
    miou = np.nanmean(ious)
    macc = np.nanmean(accs)
    oa = np.diag(hist).sum() / hist.sum() * 100
    avg_latency = np.mean(inference_times) * 1000  # ms

    print("\n" + "=" * 40)
    print("      SCALETPT EVALUATION REPORT      ")
    print("=" * 40)
    print(f"Model: PointTransformerV3 (Params: {params / 1e6:.2f}M)")
    print(f"Validation Sequences: {val_sequences}")
    print(f"Inference Latency: {avg_latency:.2f} ms/frame ({1000 / avg_latency:.1f} FPS)")
    print("-" * 40)
    print(f"Overall Accuracy (OA): {oa:.2f}%")
    print(f"Mean Accuracy (mAcc):  {macc:.2f}%")
    print(f"Mean IoU (mIoU):       {miou:.2f}%")
    print("-" * 40)

    class_names = [
        "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist",
        "motorcyclist", "road", "parking", "sidewalk", "other-ground", "building",
        "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign"
    ]

    table_data = []
    for i, name in enumerate(class_names):
        table_data.append([name, f"{ious[i]:.2f}", f"{accs[i]:.2f}"])

    print(tabulate(table_data, headers=["Class", "IoU (%)", "Acc (%)"], tablefmt="simple"))
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--labels_path', type=str, required=True)
    parser.add_argument('--sequences', type=str, nargs='+', default=None)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    main(args)