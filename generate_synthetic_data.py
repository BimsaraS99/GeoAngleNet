import os
import cv2
import numpy as np
from tqdm import tqdm

# Constants
IMAGE_SIZE = 256
NUM_SAMPLES = 1000
OUTPUT_DIR = "synthetic_dataset_complex"

def polygon_centroid(points):
    x = points[:, 0]
    y = points[:, 1]
    area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    cx = (1/(6*area)) * np.sum((x[:-1] + x[1:]) * (x[:-1]*y[1:] - x[1:]*y[:-1]))
    cy = (1/(6*area)) * np.sum((y[:-1] + y[1:]) * (x[:-1]*y[1:] - x[1:]*y[:-1]))
    return np.array([cx, cy])

def generate_complex_shape_with_directional_corners(size=256, num_points=10):
    # --- Generate polygonal shape ---
    angles = np.sort(np.random.uniform(0, 2*np.pi, num_points))
    radii = np.random.uniform(size * 0.2, size * 0.4, num_points)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.stack([x, y], axis=1)
    points = np.vstack([points, points[0]])  # Close polygon

    # --- Center the shape ---
    centroid = polygon_centroid(points)
    shift = np.array([size // 2, size // 2]) - centroid
    points_centered = points + shift
    pts = points_centered[:-1].astype(np.int32)

    # --- Draw the shape ---
    img_shape = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(img_shape, [pts], 255)

    # --- Detect polygonal corner points (inflection points) ---
    contours, _ = cv2.findContours(img_shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corner_mask = np.zeros_like(img_shape)

    if contours:
        cnt = contours[0]
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        for point in approx:
            x, y = point[0]
            cv2.circle(corner_mask, (x, y), 4, 255, thickness=-1)

    return img_shape, corner_mask

def rotate_image(img, angle, center=None):
    h, w = img.shape
    if center is None:
        center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return rotated

def create_dataset(num_samples=1000, out_dir="synthetic_dataset_complex"):
    os.makedirs(out_dir, exist_ok=True)

    for i in tqdm(range(num_samples), desc="Generating directional corner dataset"):
        img1, corner_mask1 = generate_complex_shape_with_directional_corners(
            IMAGE_SIZE, num_points=np.random.randint(6, 15)
        )

        angle = np.random.uniform(-180, 180)
        angle_mod = angle % 360

        img2 = rotate_image(img1, angle)
        corner_mask2 = rotate_image(corner_mask1, angle)

        # Save all files
        sample_dir = os.path.join(out_dir, f"sample_{i:04d}")
        os.makedirs(sample_dir, exist_ok=True)

        cv2.imwrite(os.path.join(sample_dir, "img1.png"), img1)
        cv2.imwrite(os.path.join(sample_dir, "img2.png"), img2)
        cv2.imwrite(os.path.join(sample_dir, "corner1.png"), corner_mask1)
        cv2.imwrite(os.path.join(sample_dir, "corner2.png"), corner_mask2)

        with open(os.path.join(sample_dir, "angle.txt"), "w") as f:
            f.write(str(angle_mod))

if __name__ == "__main__":
    create_dataset(NUM_SAMPLES, OUTPUT_DIR)
