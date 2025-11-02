import argparse
import os
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
        # torchrun sets the local rank env variable
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}"), local_rank, dist.get_world_size()
    else:
        # Single-machine setup
        if torch.cuda.is_available():
            # Use the first available CUDA device for single-GPU training
            return torch.device("cuda:0"), 0, 1
        elif torch.backends.mps.is_available():
            # Use MPS for Apple devices
            return torch.device("mps"), 0, 1
        else:
            # Fallback to CPU
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
        collated[key] = torch.cat(collated[key], dim=0)

    collated['batch'] = torch.cat(batch_indices, dim=0)

    return collated

def main(args):
    """The main training function, adaptable for single or distributed runs."""
    device, rank, world_size = setup_distributed()

    is_main_process = (rank == 0)
    if is_main_process:
        print(f"> Running in {'distributed' if is_distributed() else 'single-machine'} mode on device: {device}")
        print(f"> Configuration: {args}")
        if args.sequences:
            print(f"> Training on specific sequences: {args.sequences}")
        else:
            print(f"> Training on ALL sequences found in data_path")

    # Get KITTI Dataset
    full_dataset = KittiSemanticDataset(
        root_dir=args.data_path,
        labels_dir=args.labels_path,
        sequences=args.sequences
    )

    # Sampler is conditional on the environment
    sampler: Sampler
    if is_distributed():
        sampler = DistributedSampler(full_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = torch.utils.data.RandomSampler(full_dataset)

    # PyTorch DataLoader
    actual_batch_size = 1
    print(f"> DataLoader using actual batch_size: {actual_batch_size} (Ignoring args.batch_size for accumulation)")
    dataloader = DataLoader(
        full_dataset,
        batch_size=actual_batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Point Transformer V3 model
    model = PointTransformerV3(
        in_channels=4,
        enable_flash=False,

        # Reduce Channel Dimensions
        # Default: (32, 64, 128, 256, 512)
        enc_channels=(32, 64, 128, 128, 256),
        # Default: (64, 64, 128, 256)
        dec_channels=(32, 32, 64, 128),

        # Reduce Layer Depth
        # Default: (2, 2, 2, 6, 2)
        enc_depths=(2, 2, 2, 2, 2),
        # Default: (2, 2, 2, 2)
        dec_depths=(2, 2, 2, 2),

        # Reduce Head Count
        # Default: (2, 4, 8, 16, 32)
        enc_num_head=(2, 4, 8, 8, 16),
        # Default: (4, 4, 8, 16)
        dec_num_head=(4, 4, 8, 8),

        # Reduce Patch Size
        # Default: (1024, 1024, 1024, 1024, 1024)
        enc_patch_size=(256, 256, 256, 256, 256),
        # Default: (1024, 1024, 1024, 1024)
        dec_patch_size=(256, 256, 256, 256)
    ).to(device)

    num_classes = 19
    out_channels = 32
    seg_head = torch.nn.Linear(out_channels, num_classes).to(device)

    # TODO: Data parallelism
    if is_distributed():
        model = DDP(model, device_ids=[rank] if device.type == 'cuda' else None)
        seg_head = DDP(seg_head, device_ids=[rank] if device.type == 'cuda' else None)

    # Training setup
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(seg_head.parameters()),
        lr=args.learning_rate
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # GRAD ACCUM: Calculate effective batch size
    effective_batch_size = args.batch_size * args.accumulation_steps * world_size
    if is_main_process:
        print(f"> Effective Batch Size: {effective_batch_size} (batch_size * accumulation_steps * world_size)")
        print("> --- Training started ---")

    # Training loop
    for epoch in range(args.epochs):
        if is_distributed():
            sampler.set_epoch(epoch)

        optimizer.zero_grad()

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

            output = model(data_dict)
            logits = seg_head(output.feat)
            labels = data_dict["label"]

            loss = criterion(logits, labels)

            if args.accumulation_steps > 1:
                loss = loss / args.accumulation_steps

            loss.backward()

            if (i + 1) % args.accumulation_steps == 0 or (i + 1) == len(dataloader):

                optimizer.step()
                optimizer.zero_grad()

                if is_main_process and (i + 1) // args.accumulation_steps % 10 == 0:
                    print(
                        f"> Epoch: {epoch + 1}/{args.epochs} | Step: {(i + 1) // args.accumulation_steps}/{len(dataloader) // args.accumulation_steps} | Approx Loss: {loss.item() * args.accumulation_steps:.4f}")

    if is_main_process:
        print("> --- Training finished ---")

        print(f"> Saving final model weights to {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)

        model_to_save = model
        if is_distributed():
            model_to_save = model.module

        save_path = os.path.join(args.output_dir, "final_model.pt")
        torch.save(model_to_save.state_dict(), save_path)
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
    parser.add_argument('--batch_size', type=int, default=2, help="Target batch size per process (GPU) achieved via accumulation.")
    parser.add_argument('--accumulation_steps', type=int, default=4,
                        help="Number of steps to accumulate gradients over.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument('--num_workers', type=int, default=2, help="Number of workers for the DataLoader.")

    parser.add_argument('--output_dir', type=str, default="_outputs", help="Directory to save trained weights.")

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.accumulation_steps < 1:
        raise ValueError("--accumulation_steps must be >= 1")

    main(args)
