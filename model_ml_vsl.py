# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "numpy<2.0",
#     "torch",
#     "pandas"
# ]
# [tool.uv]
# exclude-newer = "2025-05-02T00:00:00Z"
# ///

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import random
import sys


class CSVDataset(Dataset):
    def __init__(self, field_csv_files, field_size=(256, 256), transform=None):
        self.field_csv_files = field_csv_files
        self.field_size = field_size
        self.transform = transform

    def __len__(self):
        return len(self.field_csv_files)

    def load_csv_as_field(self, file_path):
        df = pd.read_csv(file_path)
        field = np.random.rand(self.field_size[0], self.field_size[1]).astype(np.float32)
        mask = np.zeros(self.field_size, dtype=np.float32)
        
        for _, row in df.iterrows():
            x = int(row["s1"]) - 1
            y = int(row["s2"]) - 1
            value = float(row["z1"])
            if 0 <= x < self.field_size[0] and 0 <= y < self.field_size[1]:
                field[x, y] = value
                mask[x, y] = 1.0

        field_tensor = torch.tensor(field).unsqueeze(0)  # Add channel dimension
        mask_tensor = torch.tensor(mask).unsqueeze(0)

        return field_tensor, mask_tensor

    def __getitem__(self, idx):
        field, mask = self.load_csv_as_field(self.field_csv_files[idx])

        if self.transform:
            field = self.transform(field)
            mask = self.transform(mask)

        return field, mask

class PartialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(PartialConv2d, self).__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.mask_conv = nn.Conv2d(1, 1, kernel_size=kwargs['kernel_size'],
                                   stride=kwargs.get('stride', 1),
                                   padding=kwargs.get('padding', 0))

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        self.mask_conv.requires_grad_(False)
        self.kernel_area = kwargs['kernel_size'] * kwargs['kernel_size']

    def forward(self, input, mask):
        with torch.no_grad():
            updated_mask = self.mask_conv(mask)
            mask_ratio = self.kernel_area / (updated_mask + 1e-8)
            mask_ratio = mask_ratio * (updated_mask > 0).float()
            new_mask = (updated_mask > 0).float()

        output = self.input_conv(input * mask)
        output = output * mask_ratio

        return output, new_mask


class PConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PConvBlock, self).__init__()
        self.pconv = PartialConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, mask):
        x, mask = self.pconv(x, mask)
        x = self.relu(x)
        return x, mask


class PConvUNet(nn.Module):
    def __init__(self):
        super(PConvUNet, self).__init__()
        size = 64

        self.enc1 = PConvBlock(1, size)
        self.enc2 = PConvBlock(size, size * 2)
        self.enc3 = PConvBlock(size * 2, size * 4)
        self.enc4 = PConvBlock(size * 4, size * 8)
        self.enc5 = PConvBlock(size * 8, size * 16)

        self.dec4 = PConvBlock(size * 16 + size * 8, size * 8)
        self.dec3 = PConvBlock(size * 8 + size * 4, size * 4)
        self.dec2 = PConvBlock(size * 4 + size * 2, size * 2)
        self.dec1 = PConvBlock(size * 2 + size, size)

        self.final = PartialConv2d(size, 1, kernel_size=3, padding=1)

    def forward(self, x, mask):
        # Encoder
        e1, m1 = self.enc1(x, mask)
        e2, m2 = self.enc2(F.max_pool2d(e1, 2), F.max_pool2d(m1, 2))
        e3, m3 = self.enc3(F.max_pool2d(e2, 2), F.max_pool2d(m2, 2))
        e4, m4 = self.enc4(F.max_pool2d(e3, 2), F.max_pool2d(m3, 2))
        e5, m5 = self.enc5(F.max_pool2d(e4, 2), F.max_pool2d(m4, 2))

        # Decoder
        d4, dm4 = self.dec4(torch.cat([F.interpolate(e5, scale_factor=2, mode="nearest-exact"), e4], dim=1),
                            F.interpolate(m5, scale_factor=2, mode="nearest-exact") * m4)
        d3, dm3 = self.dec3(torch.cat([F.interpolate(e4, scale_factor=2, mode="nearest-exact"), e3], dim=1),
                            F.interpolate(m4, scale_factor=2, mode="nearest-exact") * m3)
        d2, dm2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2, mode="nearest-exact"), e2], dim=1),
                            F.interpolate(dm3, scale_factor=2, mode="nearest-exact") * m2)
        d1, dm1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2, mode="nearest-exact"), e1], dim=1),
                            F.interpolate(dm2, scale_factor=2, mode="nearest-exact") * m1)

        out, _ = self.final(d1, dm1)
        return out

def laplacian_blur(x):
    laplacian_kernel = torch.tensor(
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]], device=x.device, dtype=x.dtype
    ).view(1, 1, 3, 3)
    laplacian_kernel = laplacian_kernel.repeat(x.shape[1], 1, 1, 1)

    return F.conv2d(x, laplacian_kernel, padding=1, groups=x.shape[1])

def gaussian_blur(x, kernel_size=5, sigma=1.0):
    channels = x.shape[1]
    kernel = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, device=x.device).float()
    kernel = torch.exp(-0.5 * (kernel / sigma)**2)
    kernel = kernel / kernel.sum()
    kernel_2d = torch.outer(kernel, kernel).unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)
    kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)  # (channels, 1, k, k)

    return F.conv2d(x, kernel_2d, padding=kernel_size//2, groups=channels)

def compute_distance_transform(mask):
    """
    Compute approximate distance transform:
    mask: (B, 1, H, W), 1 = known, 0 = unknown.
    Returns distance: (B, 1, H, W), normalized to [0,1].
    """
    unknown = 1.0 - mask  # unknown = 1, known = 0

    # Apply a large blur to spread known points into unknowns
    blurred = F.avg_pool2d(unknown, kernel_size=31, stride=1, padding=15)  # large kernel
    dist = 1.0 - blurred  # closer to 1.0 = far from known

    dist = torch.clamp(dist, 0, 1)  # safety
    return dist

class AdaptiveMixedSmoothLossV2(nn.Module):
    def __init__(self, base_smoothness_weight=0.1, kernel_size=5, sigma=1.0, laplacian_weight=0.1, total_steps=10000):
        super(AdaptiveMixedSmoothLossV2, self).__init__()
        self.initial_smoothness_weight = base_smoothness_weight
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.laplacian_weight = laplacian_weight
        self.total_steps = total_steps
        self.current_step = 0

    def update_step(self, step):
        self.current_step = step

    def forward(self, pred, target, mask):
        masked_loss = (((pred - target) * mask) ** 2).sum() / mask.sum()

        pred_blurred = gaussian_blur(pred, self.kernel_size, self.sigma)
        gaussian_loss = ((pred - pred_blurred) ** 2).mean()

        pred_laplacian = laplacian_blur(pred)
        laplacian_loss = (pred_laplacian ** 2).mean()

        distance = compute_distance_transform(mask)
        # Smoothness weight decays over time
        decay = 1.0 - min(1.0, self.current_step / self.total_steps)
        smoothness_weight = self.initial_smoothness_weight * decay
        total_smoothness = (gaussian_loss + self.laplacian_weight * laplacian_loss)

        total_loss = masked_loss + (smoothness_weight * distance.mean() * total_smoothness)

        return total_loss

def train(model, dataloader, criterion, optimizer, device, epochs=10, save_path="unet_like_model.pth",
          min_loss=1e-7, loss_log_file="losses.csv"):
    model.to(device)
    model.train()

    all_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for fields, masks in dataloader:
            fields, masks = fields.to(device), masks.to(device)

            optimizer.zero_grad()
            input_fields = fields * masks
            outputs = model(input_fields, masks)
            loss = criterion(outputs, fields, masks)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        all_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.10f}")

        if avg_loss < min_loss:
            print(f"Stopping early: loss has reached below {min_loss:.1e}")
            break

    #torch.save(model.state_dict(), save_path)
    #print(f"Model saved to {save_path}")

    with open(loss_log_file, "w") as f:
        for i, loss in enumerate(all_losses, 1):
            f.write(f"{i},{loss:.10f}\n")

def predict(model, field, mask, device, output_csv_path="prediction.csv"):
    model.to(device)
    model.eval()

    with torch.no_grad():
        field = field.to(device).unsqueeze(0)
        mask = mask.to(device).unsqueeze(0)

        input_fields = field * mask
        output = model(input_fields, mask)
        output = mask.to(device) * field.to(device) + (1 - mask.to(device)) * output
        output = output.squeeze(0).cpu().numpy()

        a, b = output[0].shape
        s1 = []
        s2 = []
        for i in range(0, a):
            for j in range(0, b):
                s1.append(i)
                s2.append(j)
        z1 = output[0].flatten()

        df = pd.DataFrame({"s1": s1, "s2": s2, "z1": z1})
        df.to_csv(output_csv_path, index=False)
        print(f"Prediction saved to {output_csv_path}")

        return output

def generated_krige():
    model = PConvUNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = AdaptiveMixedSmoothLossV2(
        base_smoothness_weight=0.1,
        kernel_size=5,
        sigma=1.0,
        laplacian_weight=0.2
    )

    train_files = [sys.argv[1]]

    size = int(sys.argv[3])
    dataset = CSVDataset(train_files, field_size=(size, size))
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    train(model, train_loader, criterion, optimizer, device, epochs=250, loss_log_file=sys.argv[4])

    x = dataset[0][0]
    mask = dataset[0][1]
    pred_mask = predict(model, x, mask, device, output_csv_path=sys.argv[2])


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    generated_krige()
