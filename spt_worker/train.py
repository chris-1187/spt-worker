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
    for key in batch[0]:
        collated[key] = [b[key] for b in batch]
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
    dataloader = DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Point Transformer V3 model
    model = PointTransformerV3(in_channels=4).to(device)

    # TODO: Data parallelism
    if is_distributed():
        model = DDP(model, device_ids=[rank] if device.type == 'cuda' else None)

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    if is_main_process:
        print("> --- Training started ---")

    # Training loop
    for epoch in range(args.epochs):
        if is_distributed():
            sampler.set_epoch(epoch)

        for i, data_dict in enumerate(dataloader):
            if i == 0 and is_main_process:
                print("DEBUG - data_dict keys and types:")
                for k, v in data_dict.items():
                    print(f"  {k}: {type(v)}")
            # Move data to the target device
            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor):
                    data_dict[key] = value.to(device, non_blocking=True)
                elif isinstance(value, list):
                    data_dict[key] = [v.to(device, non_blocking=True) for v in value if isinstance(v, torch.Tensor)]
                else:
                    data_dict[key] = value

            output = model(data_dict)
            logits = output.feat
            labels = data_dict["label"]

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()  # DDP automatically averages gradients
            optimizer.step()

            if is_main_process and i % 10 == 0:
                print(f"> Epoch: {epoch + 1}/{args.epochs} | Step: {i}/{len(dataloader)} | Loss: {loss.item():.4f}")

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
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size per process (GPU).")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument('--num_workers', type=int, default=2, help="Number of workers for the DataLoader.")

    parser.add_argument('--output_dir', type=str, default="_outputs", help="Directory to save trained weights.")

    args = parser.parse_args()
    main(args)
