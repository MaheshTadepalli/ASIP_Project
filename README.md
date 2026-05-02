# Advanced Satellite Image Processing Assignment

This project performs:

1. Daubechies wavelet smoothing (`db4` or `db6`) by scaling down detail coefficients.
2. Quadtree-based region segmentation using a homogeneity condition.
3. Reconstruction of the segmented image from quadtree leaf nodes using each region average.
4. Comparison with another segmentation method using K-Means clustering.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What the UI shows

- Original input image
- Wavelet-smoothed image
- Quadtree segmented image
- Quadtree boundaries overlay
- K-Means segmented image
- Difference map between quadtree and K-Means
- Metrics like MSE, intra-region variance, region count, and segment count

## Notes

- The app keeps uploaded images in RGB format and processes all three color channels.
- Quadtree decomposition keeps splitting blocks while the average RGB channel standard deviation is above the selected threshold.
- All output images are also saved into the local `outputs` folder.
