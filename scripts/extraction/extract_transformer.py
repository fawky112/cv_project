import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from PIL import Image

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import (
    vit_b_16,
    ViT_B_16_Weights
)


SEED = 42
BATCH_SIZE = 8
NUM_WORKERS = 2

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path.home() / "Downloads" / "oxford-iiit-pet"

SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"

OUTPUT_ROOT = PROJECT_ROOT / "features" / "transformer"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


torch.manual_seed(SEED)
np.random.seed(SEED)


class PetDataset(Dataset):

    def __init__(self, split):

        self.names = np.load(
            SPLIT_ROOT / f"{split}_names.npy"
        )

        self.labels = np.load(
            SPLIT_ROOT / f"{split}_labels.npy"
        )

        self.transform = transforms.Compose([

            transforms.Resize((224, 224)),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):

        return len(self.names)

    def __getitem__(self, index):

        name = self.names[index]

        image_path = (
            DATA_ROOT / "images" / f"{name}.jpg"
        )

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            image = self.transform(image)

        label = int(self.labels[index])

        return image, label


def load_backbone():

    print("\nLoading Pretrained ViT-B/16...")

    weights = ViT_B_16_Weights.IMAGENET1K_V1

    model = vit_b_16(weights=weights)

    # Remove ImageNet classification head
    model.heads = nn.Identity()

    # Freeze all parameters
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()

    model = model.to(DEVICE)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Device: {DEVICE}")

    print(f"Total Parameters: {total_parameters:,}")

    print(f"Trainable Parameters: {trainable_parameters:,}")

    return model


@torch.inference_mode()
def extract_split(model, split):

    dataset = PetDataset(split)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    features = np.zeros(
        (len(dataset), 768),
        dtype=np.float32
    )

    labels = np.zeros(
        len(dataset),
        dtype=np.int64
    )

    current_index = 0

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    print(f"\nExtracting {split.upper()} Transformer Features...")

    for batch_index, (images, targets) in enumerate(loader):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(images)

        batch_features = outputs.cpu().numpy()

        batch_size = len(batch_features)

        features[
            current_index:current_index + batch_size
        ] = batch_features

        labels[
            current_index:current_index + batch_size
        ] = targets.numpy()

        current_index += batch_size

        if (batch_index + 1) % 20 == 0:

            print(
                f"\r{split.upper()}: "
                f"{current_index}/{len(dataset)}",
                end="",
                flush=True
            )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    print(
        f"\r{split.upper()}: "
        f"{current_index}/{len(dataset)}"
    )

    assert features.shape == (len(dataset), 768)

    assert np.all(np.isfinite(features))

    np.testing.assert_array_equal(
        labels,
        dataset.labels
    )

    np.save(
        OUTPUT_ROOT / f"transformer_{split}.npy",
        features
    )

    np.save(
        OUTPUT_ROOT / f"transformer_{split}_labels.npy",
        labels
    )

    avg_time = elapsed / len(dataset) * 1000

    print(f"Feature Matrix: {features.shape}")

    print(f"Total Time: {elapsed:.2f} sec")

    print(f"Time/Image: {avg_time:.3f} ms")

    return {
        "total_seconds": elapsed,
        "ms_per_image": avg_time
    }


def save_dictionary():

    dictionary = []

    for index in range(768):

        dictionary.append({
            "index": index,
            "name": f"vit_b16_embedding_{index}",
            "method": "pretrained_transformer",
            "computation": (
                "Dimension of final CLS token representation "
                "from frozen ViT-B/16 encoder"
            ),
            "preprocessing": (
                "Resize 224x224, RGB, ImageNet normalization"
            ),
            "resolution": [224, 224],
            "backbone": "torchvision_vit_b_16",
            "weights": "IMAGENET1K_V1"
        })

    with open(
        OUTPUT_ROOT / "transformer_feature_dictionary.json",
        "w"
    ) as file:

        json.dump(dictionary, file, indent=2)


def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 65)
    print("PRETRAINED TRANSFORMER FEATURE EXTRACTION")
    print("=" * 65)

    model = load_backbone()

    timings = {}

    for split in ["train", "val"]:

        timings[split] = extract_split(
            model,
            split
        )

    save_dictionary()

    with open(
        OUTPUT_ROOT / "transformer_extraction_times.json",
        "w"
    ) as file:

        json.dump(
            {
                "model": "ViT-B/16",
                "weights": "IMAGENET1K_V1",
                "device": str(DEVICE),
                "batch_size": BATCH_SIZE,
                "timings": timings,
                "timing_note": (
                    "End-to-end DataLoader and inference time, "
                    "including image loading and preprocessing."
                )
            },
            file,
            indent=4
        )

    print("\n" + "=" * 65)
    print("TRANSFORMER FEATURE EXTRACTION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
