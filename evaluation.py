import os
import sys
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


# ============================================================
# U-NET RESTORATION MODEL
# ============================================================

class UNetRestoration(nn.Module):

    def __init__(self):
        super().__init__()

        # -------------------------
        # Encoder
        # -------------------------

        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(64)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(128)
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(256)
        )

        self.pool = nn.MaxPool2d(2)

        # -------------------------
        # Bottleneck
        # -------------------------

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(512),
            ResidualBlock(512)
        )

        # -------------------------
        # Decoder
        # -------------------------

        self.up3 = nn.ConvTranspose2d(
            512, 256, 2, stride=2
        )

        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(256)
        )

        self.up2 = nn.ConvTranspose2d(
            256, 128, 2, stride=2
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(128)
        )

        self.up1 = nn.ConvTranspose2d(
            128, 64, 2, stride=2
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(64)
        )

        # -------------------------
        # Output
        # -------------------------

        self.output = nn.Conv2d(
            64, 1, 3, padding=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        # Bottleneck
        b = self.bottleneck(
            self.pool(e3)
        )

        # Decoder
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Output
        out = self.output(d1)

        return out


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_path, device):

    model = UNetRestoration()

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    # Your notebook saves model.state_dict(),
    # so checkpoint is directly the state dictionary.
    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


# ============================================================
# RESTORE ONE IMAGE
# ============================================================

def restore_image(model, image_path, device):

    # Load degraded 128x128 image
    noisy_lr = np.load(image_path).astype(
        np.float32
    )

    # Convert NumPy → Tensor
    x = torch.from_numpy(
        noisy_lr
    ).unsqueeze(0).unsqueeze(0)

    x = x.to(
        device,
        non_blocking=True
    )

    # --------------------------------------------------------
    # Same preprocessing used during training/inference
    # 128x128 → 256x256 using bicubic interpolation
    # --------------------------------------------------------

    x = F.interpolate(
        x,
        size=(256, 256),
        mode="bicubic",
        align_corners=False
    )

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    with torch.no_grad():
        output = model(x)

    # Convert output to NumPy
    restored = output[
        0, 0
    ].cpu().numpy()

    # Keep output in valid image range
    restored = np.clip(
        restored,
        0.0,
        1.0
    )

    return restored


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "AI-based semiconductor image "
            "restoration inference"
        )
    )

    parser.add_argument(
        "test_dir",
        type=str,
        help="Path to directory containing test .npy images"
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to directory where restored images will be saved"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Optional path to model weights. "
            "If omitted, best_model.pth must be "
            "in the same directory as this script."
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    script_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    if args.model is None:
        model_path = os.path.join(
            script_dir,
            "best_model.pth"
        )
    else:
        model_path = args.model

    # --------------------------------------------------------
    # Check inputs
    # --------------------------------------------------------

    if not os.path.isdir(args.test_dir):

        print(
            f"ERROR: Test directory not found:\n"
            f"{args.test_dir}"
        )

        sys.exit(1)

    if not os.path.isfile(model_path):

        print(
            f"ERROR: Model weights not found:\n"
            f"{model_path}"
        )

        sys.exit(1)

    # Create output directory
    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("SEMICONDUCTOR IMAGE RESTORATION")
    print("=" * 60)

    print("Test directory :", args.test_dir)
    print("Output directory:", args.output_dir)
    print("Model          :", model_path)
    print("Device         :", device)

    if torch.cuda.is_available():

        print(
            "GPU            :",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = load_model(
        model_path,
        device
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Model parameters: {total_params:,}"
    )

    # --------------------------------------------------------
    # Find test images
    # --------------------------------------------------------

    test_files = sorted([
        f for f in os.listdir(args.test_dir)
        if f.lower().endswith(".npy")
    ])

    if len(test_files) == 0:

        print(
            "\nERROR: No .npy test images found."
        )

        sys.exit(1)

    print(
        f"\nTest images found: {len(test_files)}"
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    total_time = 0.0

    print("\nStarting inference...")
    print("-" * 60)

    for i, filename in enumerate(test_files):

        input_path = os.path.join(
            args.test_dir,
            filename
        )

        # Synchronize GPU before timing
        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        restored = restore_image(
            model,
            input_path,
            device
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        inference_time = (
            end_time - start_time
        )

        total_time += inference_time

        # ----------------------------------------------------
        # Save restored output
        # ----------------------------------------------------

        output_path = os.path.join(
            args.output_dir,
            filename
        )

        np.save(
            output_path,
            restored.astype(np.float32)
        )

        # Progress
        print(
            f"[{i + 1:4d}/{len(test_files):4d}] "
            f"{filename} | "
            f"{inference_time * 1000:.2f} ms"
        )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    average_time = (
        total_time / len(test_files)
    )

    print("\n")
    print("=" * 60)
    print("INFERENCE COMPLETED")
    print("=" * 60)

    print(
        f"Images processed       : {len(test_files)}"
    )

    print(
        f"Total inference time   : "
        f"{total_time:.4f} seconds"
    )

    print(
        f"Average inference time : "
        f"{average_time * 1000:.2f} ms/image"
    )

    print(
        f"Restored outputs       : "
        f"{args.output_dir}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
