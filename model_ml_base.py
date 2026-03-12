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

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up_scale = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x2 = self.up_scale(x2)

        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x


class DownLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownLayer, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(self.pool(x))
        return x


class UpLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpLayer, self).__init__()
        self.up = Up(in_ch, out_ch)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        a = self.up(x1, x2)
        x = self.conv(a)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        size = 32

        self.conv1 = DoubleConv(in_channels, size)
        self.down1 = DownLayer(size, size * 2)
        self.down2 = DownLayer(size * 2, size * 4)
        self.down3 = DownLayer(size * 4, size * 8)
        self.down4 = DownLayer(size * 8, size * 16)
        self.up1 = UpLayer(size * 16, size * 8)
        self.up2 = UpLayer(size * 8, size * 4)
        self.up3 = UpLayer(size * 4, size * 2)
        self.up4 = UpLayer(size * 2, size)
        self.last_conv = nn.Conv2d(size, out_channels, 1)

    def forward(self, x):
        mask = x[:, 1:2, :, :]
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x1_up = self.up1(x4, x5)
        x2_up = self.up2(x3, x1_up)
        x3_up = self.up3(x2, x2_up)
        x4_up = self.up4(x1, x3_up)
        output = self.last_conv(x4_up)
        return output

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target, mask):
        loss = pred - target
        loss = ((loss * mask) ** 2).sum() / mask.sum()
        return loss

def train(model, dataloader, criterion, optimizer, device, epochs=10, save_path="model_001.pth",
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
            outputs = model(input_fields)
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
        output = model(input_fields)
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
    model = UNet(in_channels=1, out_channels=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = MaskedMSELoss()

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
