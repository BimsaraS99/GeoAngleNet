import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import os
import cv2
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # --- Encoder ---
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),  # (B, 16, 128, 128)
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # (B, 32, 64, 64)
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, 32, 32)
            nn.ReLU()
        )

        # --- Decoder ---
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (B, 32, 64, 64)
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1),  # (B, 16, 128, 128)
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),   # (B, 1, 256, 256)
            nn.Sigmoid()  # for binary mask reconstruction
        )

    def forward(self, x):
        # --- Encoding ---
        x1 = self.enc1(x)  # (B, 16, 128, 128)
        x2 = self.enc2(x1) # (B, 32, 64, 64)
        x3 = self.enc3(x2) # (B, 64, 32, 32)

        # --- Decoding ---
        d3 = self.dec3(x3)           # (B, 32, 64, 64)
        d3 = torch.cat([d3, x2], 1)  # (B, 64, 64, 64)

        d2 = self.dec2(d3)           # (B, 16, 128, 128)
        d2 = torch.cat([d2, x1], 1)  # (B, 32, 128, 128)

        out = self.dec1(d2)          # (B, 1, 256, 256)
        return out


# Reconstruct the model and optimizer exactly as before
autoencoder = Autoencoder().to(device)  # match your original config
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)

# Load checkpoint
checkpoint = torch.load("autoencoder_checkpoint.pth", map_location=torch.device(device))  # use 'cuda' if needed
autoencoder
# Restore weights and optimizer state
autoencoder.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
last_loss = checkpoint['loss']

autoencoder.eval()  # or model.train() if resuming training

print(f"Loaded model from epoch {start_epoch} with loss {last_loss:.4f}")


def extract_coordinates_from_heatmap(heatmap_tensor, threshold=0.8, center=None, degrees=True):
    heatmap = heatmap_tensor.squeeze().cpu().numpy()  # (H, W)
    H, W = heatmap.shape
    binary = (heatmap > threshold).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if center is None:
        center = (W // 2, H // 2)

    cx, cy = center
    polar_coords = []

    for i in range(1, num_labels):  # skip background
        x, y = centroids[i]
        dx, dy = x - cx, y - cy
        radius = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        if degrees:
            angle = np.degrees(angle) % 360 # Convert to degrees and normalize to [0, 360)
        polar_coords.append((radius, angle))  # Normalize angle to [0, 1]

    # Sort by radius (distance from center)
    polar_coords.sort(key=lambda x: x[0])

    return polar_coords


def match_point_sets(list1, list2, return_diff=False, max_points=16):
    arr1 = np.array(list1, dtype=np.float64)
    arr2 = np.array(list2, dtype=np.float64)

    if arr1.ndim != 2 or arr2.ndim != 2 or arr1.shape[1] != 2 or arr2.shape[1] != 2:
        arr1 = np.zeros((0, 2), dtype=np.float64)
        arr2 = np.zeros((0, 2), dtype=np.float64)

    arr1 = arr1[arr1[:, 0].argsort()]
    arr2 = arr2[arr2[:, 0].argsort()]

    if len(arr1) > len(arr2):
        arr1, arr2 = arr2, arr1

    matched1 = []
    matched2 = []
    used = set()

    for p1 in arr1:
        r1, a1 = p1
        dists = np.abs(arr2[:, 0] - r1)
        for idx in np.argsort(dists):
            if idx not in used:
                matched1.append((r1, a1))
                matched2.append((arr2[idx][0], arr2[idx][1]))
                used.add(idx)
                break

    while len(matched1) < max_points:
        matched1.append((0.0, 0.0))
        matched2.append((0.0, 0.0))

    if return_diff:
        return [(r1 - r2, abs(math.sin(math.radians(a1 - a2)))) for (r1, a1), (r2, a2) in zip(matched1, matched2)]
    else:
        return matched1, matched2
    

class Regressor(nn.Module):
    def __init__(self, autoencoder, max_points=16):
        super(Regressor, self).__init__()
        self.encoder = autoencoder
        self.max_points = max_points

        self.regressor = nn.Sequential(
            nn.Linear(max_points * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def extract_polar_coords(self, x):
        self.encoder.eval()
        with torch.no_grad():
            heatmap = self.encoder(x)[0][0]  # shape: (1, H, W)

        polar_coords = extract_coordinates_from_heatmap(
            heatmap, threshold=0.75, degrees=True
        )

        # Sort and convert angle to sin(θ)
        polar_coords = sorted(polar_coords, key=lambda p: p[0])
        polar_coords = [(r, a) for r, a in polar_coords]

        return polar_coords

    def forward(self, x1, x2):
        batch_size = x1.size(0)
        outputs = []

        for i in range(batch_size):
            coords1 = self.extract_polar_coords(x1[i].unsqueeze(0))
            coords2 = self.extract_polar_coords(x2[i].unsqueeze(0))

            # Get difference vector only
            diffs = match_point_sets(coords1, coords2, return_diff=True, max_points=self.max_points)

            # Convert to tensor
            diff_tensor = torch.tensor(diffs, dtype=torch.float32, device=x1.device).flatten()
            outputs.append(diff_tensor)

        features = torch.stack(outputs, dim=0)  # (B, max_points * 2)
        sin_pred = self.regressor(features).squeeze(1)  # (B,)
        return sin_pred