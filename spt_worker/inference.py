import argparse
import os
import json
import time
import torch
import warnings
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm

from spt_worker.dataset import KittiSemanticDataset
from spt_worker.model import PointTransformerV3

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(divide='ignore', invalid='ignore')

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

    # --- 1. Dynamic Host Configuration --- TODO: distributed
    #hostname = socket.gethostname()
    #if "host3" in hostname:
    #    CHUNK_SIZE = 10000
    #    print(f"> [Config] Detected Host3. Chunk size set to {CHUNK_SIZE}")
    #else:
    #    CHUNK_SIZE = 40000
    #    print(f"> [Config] Detected High-Perf Node. Chunk size set to {CHUNK_SIZE}")

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
        sampling_strategy=args.sampling_strategy
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

    with torch.no_grad():
        for i, data_dict in tqdm(enumerate(dataloader), total=len(dataloader)):
            full_coord = data_dict['coord'].to(device)
            full_feat = data_dict['feat'].to(device)
            full_label = data_dict['label'].to(device)

            total_points = full_coord.shape[0]
            full_logits = torch.zeros((total_points, num_classes), device=device)
            full_count = torch.zeros((total_points), device=device)

            start = 0
            torch.cuda.synchronize()
            t_start = time.time()

            while start < total_points:
                end = min(start + chunk_size, total_points)
                chunk_coord = full_coord[start:end]
                chunk_feat = full_feat[start:end]
                chunk_batch_idx = torch.zeros(chunk_coord.shape[0], dtype=torch.long, device=device)

                chunk_input = {
                    "coord": chunk_coord,
                    "feat": chunk_feat,
                    "batch": chunk_batch_idx,
                    "grid_size": 0.01
                }

                output = model(chunk_input)
                logits = seg_head(output.feat)

                full_logits[start:end] += logits
                full_count[start:end] += 1.0

                if end == total_points:
                    break
                start += stride

            torch.cuda.synchronize()
            inference_times.append(time.time() - t_start)

            final_preds = (full_logits / full_count.unsqueeze(-1)).argmax(dim=1)
            preds_np = final_preds.cpu().numpy()
            labels_np = full_label.cpu().numpy()

            valid_mask = labels_np != -1
            hist += fast_hist(preds_np[valid_mask], labels_np[valid_mask], num_classes)

            # Map 0-18 indices back to standard SemanticKITTI uint32 IDs
            inverse_label_map = {
                0: 10, 1: 11, 2: 15, 3: 18, 4: 20, 5: 30, 6: 31, 7: 32,
                8: 40, 9: 44, 10: 48, 11: 49, 12: 50, 13: 51, 14: 70,
                15: 71, 16: 72, 17: 80, 18: 81
            }
            preds_kitti = np.vectorize(inverse_label_map.get)(preds_np).astype(np.uint32)

            original_path = data_dict['file_path'][0]
            seq_id = original_path.split('/')[-3]
            frame_name = original_path.split('/')[-1].replace('.bin', '.label')
            save_dir = os.path.join(inference_dir, "predictions", seq_id)
            os.makedirs(save_dir, exist_ok=True)

            preds_kitti.tofile(os.path.join(save_dir, frame_name))

    # Final Metrics
    ious = per_class_iu(hist) * 100
    accs = per_class_acc(hist) * 100

    # Construct JSON payload
    inference_metrics = {
        "event": "evaluation_completed",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "chunk_size": chunk_size,
            "overlap_size": args.overlap_size,
            "sampling_strategy": args.sampling_strategy,
            "sequences": args.sequences
        },
        "results": {
            "overall_accuracy": round(np.diag(hist).sum() / hist.sum() * 100, 4),
            "mean_accuracy": round(np.nanmean(accs), 4),
            "mean_iou": round(np.nanmean(ious), 4),
            "average_latency_ms": round(np.mean(inference_times) * 1000, 4),
            "per_class_iou": {str(i): round(val, 4) for i, val in enumerate(ious)},
            "per_class_acc": {str(i): round(val, 4) for i, val in enumerate(accs)}
        }
    }

    metrics_file = os.path.join(inference_dir, "evaluation_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(inference_metrics, f, indent=4)

    print(f"> [Inference] Evaluation complete. Metrics appended to {metrics_file}")


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
    args = parser.parse_args()
    main(args)