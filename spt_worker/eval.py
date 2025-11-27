import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from spt_worker.dataset import KittiSemanticDataset
from spt_worker.model import PointTransformerV3


def fast_hist(pred, label, n):
    """
    Calculates the confusion matrix (Intersection and Union counts)
    """
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2
    ).reshape(n, n)


def per_class_iu(hist):
    """
    Calculates Intersection over Union (IoU) for each class
    """
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"> Running evaluation on device: {device}")

    # Dataset
    # keep max_points same as training to ensure it fits in GPU
    val_dataset = KittiSemanticDataset(
        root_dir=args.data_path,
        labels_dir=args.labels_path,
        sequences=args.sequences,
        max_points=5000
    )

    dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )

    # Model Setup
    # Must match training configuration
    out_channels = 32
    num_classes = 19

    model = PointTransformerV3(
        in_channels=4,
        enable_flash=False,
        # Reduce Channel Dimensions
        enc_channels=(32, 64, 128, 128, 256),
        dec_channels=(32, 32, 64, 128),
        enc_depths=(2, 2, 2, 2, 2),
        dec_depths=(2, 2, 2, 2),
        enc_num_head=(2, 4, 8, 8, 16),
        dec_num_head=(4, 4, 8, 8),
        enc_patch_size=(256, 256, 256, 256, 256),
        dec_patch_size=(256, 256, 256, 256)
    ).to(device)

    seg_head = torch.nn.Linear(out_channels, num_classes).to(device)

    # Load Checkpoint
    print(f"> Loading weights from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    # Handle the dictionary format
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        seg_head.load_state_dict(checkpoint['seg_head_state_dict'])
    else:
        # Fallback for old checkpoints
        print("! WARNING: Old checkpoint format detected. SegHead weights might be missing/random.")
        model.load_state_dict(checkpoint)

    model.eval()
    seg_head.eval()

    # Evaluation Loop
    hist = np.zeros((num_classes, num_classes))
    print("> Starting inference...")

    with torch.no_grad():
        for i, data_dict in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Prepare data
            data_dict['grid_size'] = 0.01
            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor):
                    data_dict[key] = value.to(device, non_blocking=True)
                else:
                    data_dict[key] = value

            # Forward Pass
            # Handle the collate/batching manually if not using the train collate_fn
            # But since batch_size=1, pass data directly if we add the 'batch' key
            if 'batch' not in data_dict:
                data_dict['batch'] = torch.zeros(data_dict['coord'].shape[0], dtype=torch.long, device=device)

            output = model(data_dict)
            logits = seg_head(output.feat)

            # Get predictions
            preds = logits.argmax(dim=1)
            labels = data_dict['label']

            # Accumulate stats
            # Move to CPU for numpy calculation
            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()

            # Filter out ignore_index (-1)
            valid_mask = labels_np != -1
            hist += fast_hist(preds_np[valid_mask], labels_np[valid_mask], num_classes)

    # Calculate Metrics
    ious = per_class_iu(hist) * 100
    miou = np.nanmean(ious)

    print("\n> --- Evaluation Results ---")
    print(f"> mIoU: {miou:.2f}%")
    print("> Per-class IoU:")

    # Mapping indices back to names (Optional, based on LABEL_MAP)
    # This list must match the order 0-18 in the LABEL_MAP values
    class_names = [
        "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist",
        "motorcyclist", "road", "parking", "sidewalk", "other-ground", "building",
        "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign"
    ]

    for i, iou in enumerate(ious):
        name = class_names[i] if i < len(class_names) else str(i)
        print(f"  {name}: {iou:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--labels_path', type=str, required=True)
    parser.add_argument('--sequences', type=str, nargs='+', default=['04'])
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    main(args)