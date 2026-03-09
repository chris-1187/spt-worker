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

    with torch.no_grad():
        for i, data_dict in tqdm(enumerate(dataloader), total=len(dataloader)):
            full_coord = data_dict['coord'].to(device)
            full_feat = data_dict['feat'].to(device)
            full_label = data_dict['label'].to(device)

            total_points = full_coord.shape[0]
            full_logits = torch.zeros((total_points, num_classes), device=device) # Raw, unnormalized network outputs (logits)
            full_count = torch.zeros((total_points), device=device) # Count of evaluations per point

            torch.cuda.synchronize()
            t_start = time.time()

            if args.sampling_strategy == 'hilbert':
                # ---------------------------------------------------------
                # HILBERT: 1D sliding window over sorted points
                # ---------------------------------------------------------
                start = 0
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

            elif args.sampling_strategy == 'knn':
                # ---------------------------------------------------------
                # K-NEAREST NEIGHBORS: Minimum coverage spherical algorithm
                # ---------------------------------------------------------
                coverage_count = torch.zeros(total_points, device=device)

                while coverage_count.min() == 0:
                    min_cov = coverage_count.min()
                    candidate_indices = torch.nonzero(coverage_count == min_cov).squeeze(1)

                    if candidate_indices.dim() == 0 or candidate_indices.numel() == 1:
                        seed_idx = candidate_indices
                    else:
                        rand_pos = torch.randint(0, len(candidate_indices), (1,), device=device)
                        seed_idx = candidate_indices[rand_pos].squeeze()

                    seed_point = full_coord[seed_idx]
                    dists = torch.sum((full_coord - seed_point) ** 2, dim=1)

                    k = min(chunk_size, total_points)
                    _, knn_indices = torch.topk(dists, k, largest=False)

                    chunk_coord = full_coord[knn_indices]
                    chunk_feat = full_feat[knn_indices]
                    chunk_batch_idx = torch.zeros(chunk_coord.shape[0], dtype=torch.long, device=device)

                    chunk_input = {
                        "coord": chunk_coord,
                        "feat": chunk_feat,
                        "batch": chunk_batch_idx,
                        "grid_size": 0.01
                    }

                    # forward pass
                    output = model(chunk_input)
                    logits = seg_head(output.feat)

                    full_logits[knn_indices] += logits
                    full_count[knn_indices] += 1.0

                    # mark as covered
                    coverage_count[knn_indices] += 1.0

            torch.cuda.synchronize()
            inference_times.append(time.time() - t_start)

            final_preds = (full_logits / full_count.unsqueeze(-1)).argmax(dim=1) # fusion
            preds_np = final_preds.cpu().numpy()
            labels_np = full_label.cpu().numpy()

            valid_mask = labels_np != -1
            hist += fast_hist(preds_np[valid_mask], labels_np[valid_mask], num_classes)

            # Map 0-18 indices back to standard SemanticKITTI ids
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