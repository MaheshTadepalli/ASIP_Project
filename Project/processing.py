import io
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pywt
from PIL import Image
from sklearn.cluster import KMeans


@dataclass
class QuadNode:
    row: int
    col: int
    height: int
    width: int
    mean_value: float
    std_value: float


def load_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(file_bytes)).convert("L")
    return np.array(image, dtype=np.uint8)


def wavelet_smooth(image_array: np.ndarray, wavelet_name: str, level: int, detail_scale: float) -> np.ndarray:
    image_float = image_array.astype(np.float32)
    coeffs = pywt.wavedec2(image_float, wavelet=wavelet_name, level=level)

    adjusted_coeffs = [coeffs[0]]
    for details in coeffs[1:]:
        adjusted_coeffs.append(tuple(detail_scale * component for component in details))

    reconstructed = pywt.waverec2(adjusted_coeffs, wavelet=wavelet_name)
    reconstructed = reconstructed[: image_array.shape[0], : image_array.shape[1]]
    reconstructed = np.clip(reconstructed, 0, 255)
    return reconstructed.astype(np.uint8)


def split_condition(block: np.ndarray, threshold: float, min_block_size: int) -> bool:
    return (
        block.shape[0] >= 2 * min_block_size
        and block.shape[1] >= 2 * min_block_size
        and np.std(block) > threshold
    )


def quadtree_decompose(
    image_array: np.ndarray,
    threshold: float,
    min_block_size: int,
    row: int = 0,
    col: int = 0,
) -> List[QuadNode]:
    height, width = image_array.shape

    if height == 0 or width == 0:
        return []

    if not split_condition(image_array, threshold, min_block_size) or height < 2 or width < 2:
        return [
            QuadNode(
                row=row,
                col=col,
                height=height,
                width=width,
                mean_value=float(np.mean(image_array)),
                std_value=float(np.std(image_array)),
            )
        ]

    half_h = height // 2
    half_w = width // 2

    blocks = [
        (image_array[:half_h, :half_w], row, col),
        (image_array[:half_h, half_w:], row, col + half_w),
        (image_array[half_h:, :half_w], row + half_h, col),
        (image_array[half_h:, half_w:], row + half_h, col + half_w),
    ]

    nodes: List[QuadNode] = []
    for block, block_row, block_col in blocks:
        nodes.extend(quadtree_decompose(block, threshold, min_block_size, block_row, block_col))
    return nodes


def build_segmented_image(shape: Tuple[int, int], nodes: List[QuadNode]) -> Tuple[np.ndarray, np.ndarray]:
    segmented = np.zeros(shape, dtype=np.uint8)
    labels = np.zeros(shape, dtype=np.int32)

    for index, node in enumerate(nodes, start=1):
        value = int(np.clip(round(node.mean_value), 0, 255))
        segmented[node.row : node.row + node.height, node.col : node.col + node.width] = value
        labels[node.row : node.row + node.height, node.col : node.col + node.width] = index

    return segmented, labels


def overlay_boundaries(image_array: np.ndarray, labels: np.ndarray) -> np.ndarray:
    rgb = np.stack([image_array] * 3, axis=-1)
    boundary_mask = np.zeros(labels.shape, dtype=bool)
    boundary_mask[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary_mask[1:, :] |= labels[1:, :] != labels[:-1, :]
    rgb[boundary_mask] = np.array([255, 99, 71], dtype=np.uint8)
    return rgb


def kmeans_segment(image_array: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, np.ndarray]:
    flattened = image_array.reshape(-1, 1).astype(np.float32)
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = model.fit_predict(flattened)
    centers = model.cluster_centers_.flatten()
    segmented = centers[labels].reshape(image_array.shape)
    segmented = np.clip(segmented, 0, 255).astype(np.uint8)
    return segmented, labels.reshape(image_array.shape)


def compute_mean_region_variance(image_array: np.ndarray, labels: np.ndarray) -> float:
    variances: List[float] = []
    for label in np.unique(labels):
        region_pixels = image_array[labels == label]
        if region_pixels.size == 0:
            continue
        variances.append(float(np.var(region_pixels)))
    if not variances:
        return 0.0
    return float(np.mean(variances))


def compare_segmentations(results: Dict[str, np.ndarray]) -> Dict[str, float]:
    smoothed = results["smoothed_image"]
    quadtree = results["quadtree_segmented"]
    kmeans = results["kmeans_segmented"]

    return {
        "quadtree_mse": float(np.mean((smoothed.astype(np.float32) - quadtree.astype(np.float32)) ** 2)),
        "kmeans_mse": float(np.mean((smoothed.astype(np.float32) - kmeans.astype(np.float32)) ** 2)),
        "quadtree_regions": int(len(results["quadtree_nodes"])),
        "kmeans_segments": int(results["kmeans_clusters"]),
        "quadtree_intra_region_variance": compute_mean_region_variance(
            smoothed, results["quadtree_labels"]
        ),
        "kmeans_intra_region_variance": compute_mean_region_variance(
            smoothed, results["kmeans_labels"]
        ),
    }


def difference_heatmap(quadtree_image: np.ndarray, kmeans_image: np.ndarray) -> np.ndarray:
    difference = np.abs(quadtree_image.astype(np.int16) - kmeans_image.astype(np.int16)).astype(np.uint8)
    max_difference = float(np.max(difference))
    if max_difference == 0:
        normalized = np.zeros_like(difference, dtype=np.float32)
    else:
        normalized = difference.astype(np.float32) / max_difference
    colored = plt.cm.inferno(normalized)[..., :3]
    return (colored * 255).astype(np.uint8)


def create_comparison_figure(results: Dict[str, np.ndarray], metrics: Dict[str, float]):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor("#ffffff")

    panels = [
        ("Original", results["original_image"], "gray"),
        ("Wavelet Smoothed", results["smoothed_image"], "gray"),
        ("Quadtree Segmented", results["quadtree_segmented"], "gray"),
        ("Quadtree Boundaries", results["quadtree_overlay"], None),
        ("K-Means Segmented", results["kmeans_segmented"], "gray"),
        ("Difference Map", results["difference_map"], None),
    ]

    for ax, (title, image, cmap) in zip(axes.ravel(), panels):
        if cmap:
            ax.imshow(image, cmap=cmap)
        else:
            ax.imshow(image)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.suptitle(
        (
            f"Quadtree MSE: {metrics['quadtree_mse']:.2f} | "
            f"K-Means MSE: {metrics['kmeans_mse']:.2f}"
        ),
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout()
    return fig


def run_pipeline(
    image_array: np.ndarray,
    wavelet_name: str,
    level: int,
    detail_scale: float,
    homogeneity_threshold: float,
    min_block_size: int,
    k_clusters: int,
) -> Dict[str, np.ndarray]:
    smoothed_image = wavelet_smooth(image_array, wavelet_name, level, detail_scale)
    quadtree_nodes = quadtree_decompose(smoothed_image, homogeneity_threshold, min_block_size)
    quadtree_segmented, quadtree_labels = build_segmented_image(smoothed_image.shape, quadtree_nodes)
    quadtree_overlay = overlay_boundaries(smoothed_image, quadtree_labels)
    kmeans_segmented, kmeans_labels = kmeans_segment(smoothed_image, k_clusters)
    difference_map = difference_heatmap(quadtree_segmented, kmeans_segmented)

    return {
        "original_image": image_array,
        "smoothed_image": smoothed_image,
        "quadtree_segmented": quadtree_segmented,
        "quadtree_labels": quadtree_labels,
        "quadtree_overlay": quadtree_overlay,
        "quadtree_nodes": quadtree_nodes,
        "kmeans_segmented": kmeans_segmented,
        "kmeans_labels": kmeans_labels,
        "kmeans_clusters": k_clusters,
        "difference_map": difference_map,
    }
