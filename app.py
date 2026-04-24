from pathlib import Path

import streamlit as st
from PIL import Image

from processing import compare_segmentations, load_image_from_bytes, run_pipeline


st.set_page_config(
    page_title="Wavelet Smoothing and Quadtree Segmentation",
    page_icon="SAT",
    layout="wide",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_metrics_table(metrics: dict) -> list[dict[str, str]]:
    return [
        {
            "Metric": "MSE",
            "Quadtree": f"{metrics['quadtree_mse']:.2f}",
            "K-Means": f"{metrics['kmeans_mse']:.2f}",
        },
        {
            "Metric": "Intra-region Variance",
            "Quadtree": f"{metrics['quadtree_intra_region_variance']:.2f}",
            "K-Means": f"{metrics['kmeans_intra_region_variance']:.2f}",
        },
        {
            "Metric": "Number of Segments",
            "Quadtree": str(metrics["quadtree_regions"]),
            "K-Means": str(metrics["kmeans_segments"]),
        },
    ]


def main() -> None:
    inject_styles()
    st.title("Wavelet Smoothing and Quadtree Segmentation")

    with st.sidebar:
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        )
        wavelet_name = st.selectbox("Wavelet", ["db4", "db6"], index=0)
        level = st.slider("Decomposition Level", min_value=1, max_value=4, value=2)
        detail_scale = st.slider(
            "Detail Scaling Factor",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.05,
        )
        homogeneity_threshold = st.slider(
            "Quadtree Threshold",
            min_value=1.0,
            max_value=40.0,
            value=10.0,
            step=0.5,
        )
        min_block_size = st.select_slider(
            "Minimum Block Size",
            options=[2, 4, 8, 16, 32],
            value=8,
        )
        k_clusters = st.slider("K-Means Clusters", min_value=2, max_value=10, value=4)

    if not uploaded_file:
        st.info("Upload an image.")
        return

    image_array = load_image_from_bytes(uploaded_file.getvalue())
    results = run_pipeline(
        image_array=image_array,
        wavelet_name=wavelet_name,
        level=level,
        detail_scale=detail_scale,
        homogeneity_threshold=homogeneity_threshold,
        min_block_size=min_block_size,
        k_clusters=k_clusters,
    )
    metrics = compare_segmentations(results)

    st.subheader("Metrics")
    st.table(build_metrics_table(metrics))

    st.subheader("Images")
    row_1 = st.columns(2)
    with row_1[0]:
        st.image(results["original_image"], caption="Original Image", use_container_width=True)
    with row_1[1]:
        st.image(results["smoothed_image"], caption="Wavelet Smoothed Image", use_container_width=True)

    row_2 = st.columns(2)
    with row_2[0]:
        st.image(results["quadtree_segmented"], caption="Quadtree Segmented Image", use_container_width=True)
    with row_2[1]:
        st.image(results["kmeans_segmented"], caption="K-Means Segmented Image", use_container_width=True)

    row_3 = st.columns(2)
    with row_3[0]:
        st.image(results["quadtree_overlay"], caption="Quadtree Boundaries", use_container_width=True)
    with row_3[1]:
        st.image(results["difference_map"], caption="Difference Map", use_container_width=True)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    Image.fromarray(results["smoothed_image"]).save(output_dir / "smoothed.png")
    Image.fromarray(results["quadtree_segmented"]).save(output_dir / "quadtree_segmented.png")
    Image.fromarray(results["quadtree_overlay"]).save(output_dir / "quadtree_overlay.png")
    Image.fromarray(results["kmeans_segmented"]).save(output_dir / "kmeans_segmented.png")
    Image.fromarray(results["difference_map"]).save(output_dir / "difference_map.png")


if __name__ == "__main__":
    main()
