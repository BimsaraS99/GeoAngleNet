import torch
import torch.nn as nn
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from classes import Autoencoder, Regressor  # Ensure model definition is importable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Load Models -----
autoencoder = Autoencoder().to(device)
regressor = Regressor(autoencoder).to(device)

# Load saved weights
ae_ckpt = torch.load("trained_models/autoencoder_checkpoint.pth", map_location=device)
autoencoder.load_state_dict(ae_ckpt["model_state_dict"])
autoencoder.eval()

reg_ckpt = torch.load("trained_models/regressor_checkpoint.pth", map_location=device)
regressor.load_state_dict(reg_ckpt["model_state_dict"])
regressor.eval()

print(f"Autoencoder @ Epoch {ae_ckpt['epoch']} | Regressor @ Epoch {reg_ckpt['epoch']}")
print(f"Regressor Loss: {reg_ckpt['loss']:.4f}")


# ----- Inference Utilities -----
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # (1, H, W)
    img = np.expand_dims(img, axis=0)  # (1, 1, H, W)
    return torch.tensor(img, dtype=torch.float32).to(device)


def predict_angle_from_images(path1, path2, show=True):
    img1 = preprocess_image(path1)
    img2 = preprocess_image(path2)

    with torch.no_grad():
        sin_pred = regressor(img1, img2).item()

    # Clamp and convert
    sin_pred = max(-1.0, min(1.0, sin_pred))
    angle_rad = math.asin(sin_pred)
    angle_deg = math.degrees(angle_rad) % 360

    print(f"Predicted sin(angle): {sin_pred:.4f}")
    print(f"Predicted angle:     {angle_deg:.2f} degrees.")

    if show:
        img1_np = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        img2_np = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(img1_np, cmap='gray')
        axs[0].set_title("Image 1")
        axs[0].axis('off')

        axs[1].imshow(img2_np, cmap='gray')
        axs[1].set_title("Image 2")
        axs[1].axis('off')

        plt.suptitle(f"Predicted Angle: {angle_deg:.2f}°", fontsize=14)
        plt.tight_layout()
        plt.show()

    return angle_deg


# ----- Main Example -----
if __name__ == "__main__":
    img_path1 = "synthetic_dataset_complex/sample_0004/img1.png"
    img_path2 = "synthetic_dataset_complex/sample_0004/img2.png"

    predicted_angle = predict_angle_from_images(img_path1, img_path2, show=True)
