import argparse
import os
import json
import time
import platform
from datetime import datetime
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

# modules
from .dataset import KittiSemanticDataset
from .model import PointTransformerV3

IGNORE_INDEX = -1

def is_distributed():
    """Checks if the script is running in a distributed environment."""
    return "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1


def setup_distributed():
    """Initializes the distributed process group."""
    if is_distributed():
        dist.init_process_group("nccl")

        local_rank = int(os.environ["LOCAL_RANK"])

        global_rank = dist.get_rank()

        torch.cuda.set_device(local_rank)

        return torch.device(f"cuda:{local_rank}"), global_rank, dist.get_world_size()
    else:
        # Single-machine setup
        if torch.cuda.is_available():
            return torch.device("cuda:0"), 0, 1
        elif torch.backends.mps.is_available():
            return torch.device("mps"), 0, 1
        else:
            return torch.device("cpu"), 0, 1


def cleanup():
    """Cleans up the distributed process group."""
    if is_distributed():
        dist.destroy_process_group()


def collate_fn(batch):
    """Custom collate function to handle variable-size point clouds."""
    collated = {}
    batch_indices = []

    # Iterate over each sample in the batch
    for i, sample in enumerate(batch):
        num_points = sample['coord'].shape[0]

        batch_indices.append(torch.full((num_points,), i, dtype=torch.long))

        for key, value in sample.items():
            if key not in collated:
                collated[key] = []
            collated[key].append(value)

    # Concatenate all tensors for each key
    for key in collated:
        if isinstance(collated[key][0], torch.Tensor):
            collated[key] = torch.cat(collated[key], dim=0)

    collated['batch'] = torch.cat(batch_indices, dim=0)

    return collated

def main(args):
    """The main training function, adaptable for single or distributed runs."""
    total_start_time = time.time()

    device, rank, world_size = setup_distributed()
    print(f"> [DEBUG] Global Rank: {rank} | Local Device: {device} | World Size: {world_size}")

    # ------------------------------------------------------------------
    # MODEL CONFIG
    # ------------------------------------------------------------------
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

    # Logs
    is_main_process = (rank == 0)
    log_dir = args.output_dir
    if is_main_process:
        os.makedirs(log_dir, exist_ok=True)

        effective_batch_size = args.batch_size * args.accumulation_steps * world_size
        run_mode = "distributed" if is_distributed() else "single-node"

        run_config = {
            "meta": {
                "timestamp": datetime.now().isoformat(),
                "run_mode": run_mode,
                "node_hostname": platform.node()
            },
            "arguments": vars(args),
            "model_architecture": model_config,
            "training_dynamics": {
                "effective_batch_size": effective_batch_size,
                "world_size": world_size,
                "accumulation_steps": args.accumulation_steps,
                "local_batch_size": 1
            },
            "environment": {
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU"
            }
        }

        if not args.resume:
            with open(os.path.join(log_dir, "run_config.json"), "w") as f:
                json.dump(run_config, f, indent=4)
            print(f"> [Logging] Configuration saved to: {log_dir}/run_config.json")

        print(f"> [Logging] Configuration saved to: {log_dir}/run_config.json")
        print(f"> [Logging] Metrics will be saved to: {log_dir}/metrics.jsonl")
        print(f"> Running in '{run_mode}' mode on device: {device}")
        print(f"> Sampling Strategy: {args.sampling_strategy.upper()}")
        print(f"> Configuration: {args}")
        if args.sequences:
            print(f"> Training on specific sequences: {args.sequences}")
        else:
            print(f"> Training on ALL sequences found in data_path")

    # Get KITTI Dataset
    full_dataset = KittiSemanticDataset(
        root_dir=args.data_path,
        labels_dir=args.labels_path,
        sequences=args.sequences,
        training=True,
        sampling_strategy=args.sampling_strategy
    )

    # Sampler is conditional on the environment
    sampler: Sampler
    if is_distributed():
        sampler = DistributedSampler(full_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = torch.utils.data.RandomSampler(full_dataset)

    # PyTorch DataLoader
    dataloader = DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Point Transformer V3 model
    model = PointTransformerV3(**model_config).to(device)

    num_classes = 19
    out_channels = 32
    seg_head = torch.nn.Linear(out_channels, num_classes).to(device)

    if is_distributed():
        model = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None)
        seg_head = DDP(seg_head, device_ids=[device.index] if device.type == 'cuda' else None)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(seg_head.parameters()),
        lr=args.learning_rate
    )

    step_size = int(args.epochs * 0.3)
    if step_size < 1: step_size = 1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        if is_main_process:
            print(f"> [Resume] Loading checkpoint from {args.resume}")

        # Load the checkpoint
        checkpoint = torch.load(args.resume, map_location=device)

        # Unwrap DDP if necessary to load weights
        model_to_load = model.module if is_distributed() else model
        seg_head_to_load = seg_head.module if is_distributed() else seg_head

        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        seg_head_to_load.load_state_dict(checkpoint['seg_head_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1

        if is_main_process:
            print(f"> [Resume] Successfully loaded. Resuming training from epoch {start_epoch + 1}")

    # GRAD ACCUM: Calculate effective batch size
    effective_batch_size = args.batch_size * args.accumulation_steps * world_size
    if is_main_process:
        print(f"> Effective Batch Size: {effective_batch_size} (batch_size * accumulation_steps * world_size)")
        print(f"> Scheduler enabled. LR will decay every {step_size} epochs.")
        print("> --- Training started ---")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        if is_distributed():
            sampler.set_epoch(epoch)

        optimizer.zero_grad()
        epoch_loss_sum = 0.0
        num_batches = 0

        for i, data_dict in enumerate(dataloader):
            data_dict['grid_size'] = 0.01

            # Move data to the target device
            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor):
                    data_dict[key] = value.to(device, non_blocking=True)
                else:
                    data_dict[key] = value

            if is_main_process:
                print(f">>> DEBUG: Points in batch {i}: {data_dict['coord'].shape[0]}")

            # Forward Pass
            output = model(data_dict)
            logits = seg_head(output.feat)
            labels = data_dict["label"]
            loss = criterion(logits, labels)

            if torch.isnan(loss):
                print(f"> [Warning] NaN loss on Rank {rank} at batch {i}. Injecting connected zero-loss.")
                loss = (logits * 0.0).sum()

            # Tack loss of current batch to average later
            epoch_loss_sum += loss.item()
            num_batches += 1

            if args.accumulation_steps > 1:
                loss = loss / args.accumulation_steps

            loss.backward()

            if (i + 1) % args.accumulation_steps == 0 or (i + 1) == len(dataloader):

                optimizer.step()
                optimizer.zero_grad()

                if is_main_process and (i + 1) // args.accumulation_steps % 10 == 0:
                    current_loss = loss.item() * args.accumulation_steps
                    print(
                        f"> Epoch: {epoch + 1}/{args.epochs} | Step: {(i + 1) // args.accumulation_steps} | Approx Loss: {current_loss:.4f}")

        scheduler.step()

        if rank == 0:
            epoch_duration = time.time() - epoch_start_time
            avg_train_loss = epoch_loss_sum / num_batches if num_batches > 0 else 0
            current_lr = scheduler.get_last_lr()[0]  # Log the new LR
            print(f"> [Info] Epoch {epoch + 1} LR: {current_lr:.6f}")

            stats = {
                "epoch": epoch + 1,
                "timestamp": datetime.now().isoformat(),
                "epoch_duration_sec": round(epoch_duration, 2),
                "learning_rate": current_lr,
                "train_loss": avg_train_loss,
                # val_mIoU: after validation loop
            }

            with open(os.path.join(log_dir, "metrics.jsonl"), "a") as f:
                f.write(json.dumps(stats) + "\n")

            print(f"> [Logging] Epoch {epoch + 1} finished. Avg Loss: {avg_train_loss:.4f}")

            if (epoch + 1) % 5 == 0:
                print(f"> [Checkpoint] Saving intermediate model at Epoch {epoch + 1}...")

                model_to_save = model.module if is_distributed() else model
                seg_head_to_save = seg_head.module if is_distributed() else seg_head

                checkpoint = {
                    'model_state_dict': model_to_save.state_dict(),
                    'seg_head_state_dict': seg_head_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'config': run_config
                }

                # Update path to use the weights subdirectory
                weights_dir = os.path.join(args.output_dir, "weights")
                os.makedirs(weights_dir, exist_ok=True)  # Failsafe
                save_filename = f"model_weights_epoch_{epoch + 1}.pt"
                save_path = os.path.join(weights_dir, save_filename)
                torch.save(checkpoint, save_path)
                print(f"> [Checkpoint] Saved to {save_path}")

    if is_main_process:
        total_duration = time.time() - total_start_time

        final_stats = {
            "status": "finished",
            "total_duration_sec": round(total_duration, 2),
            "total_duration_min": round(total_duration / 60, 2),
            "timestamp": datetime.now().isoformat()
        }
        with open(os.path.join(log_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(final_stats) + "\n")

        print(f"> --- Training finished in {total_duration/60:.2f} minutes ---")
        print(f"> Saving final model weights to {args.output_dir}")

        model_to_save = model.module if is_distributed() else model
        seg_head_to_save = seg_head.module if is_distributed() else seg_head

        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'seg_head_state_dict': seg_head_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'config': run_config
        }

        weights_dir = os.path.join(args.output_dir, "weights")
        save_path = os.path.join(weights_dir, "model_weights.pt")
        torch.save(checkpoint, save_path)
        print(f"> Model saved to {save_path}")

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ScalePT-Worker Training Script")

    # Data and Model Paths
    parser.add_argument('--data_path', type=str, required=True, help="Root path to the datasets point clouds.")
    parser.add_argument('--labels_path', type=str, required=True, help="Root path to the dataset's labels.")
    parser.add_argument('--sequences', type=str, nargs='+', default=None,
                        help="List of sequence IDs to use (e.g., --sequences 00 01 02). If not set, all are used.")

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=1, help="Target batch size per process (GPU) achieved via accumulation.")
    parser.add_argument('--accumulation_steps', type=int, default=4,
                        help="Number of steps to accumulate gradients over.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument('--num_workers', type=int, default=2, help="Number of workers for the DataLoader.")

    parser.add_argument('--output_dir', type=str, default="_outputs", help="Directory to save trained weights, metrics, and logs.")
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to checkpoint .pt file to resume training from.")
    parser.add_argument('--sampling_strategy', type=str, default='block',
                        choices=['block', 'hilbert', 'fps_knn', 'voxel_knn'],
                        help="Strategy to sample point chunks: 'block', 'hilbert', 'fps_knn', 'voxel_knn'.")

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.accumulation_steps < 1:
        raise ValueError("--accumulation_steps must be >= 1")

    main(args)
