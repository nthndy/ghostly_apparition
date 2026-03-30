# ghostly_apparition

### a.k.a. napari time projection

Renders a single-cell time-projection animation using napari. Each timepoint is stacked along the Z axis to build a 3D temporal tower, then screenshotted frame-by-frame and compiled into an MP4.

Available as a **notebook** (`time_projection.ipynb`) for interactive use, or a **script** (`time_projection.py`) for headless / pipeline use.

## Usage

### Notebook

1. Clone / download the repo
2. Install dependencies (into an existing napari environment):
```bash
pip install zarr pandas scipy scikit-image imageio imageio-ffmpeg tqdm
```
3. Open `time_projection.ipynb` and fill in the paths and parameters in **Cell 2 (Configuration)**
4. Run all cells top to bottom

### Script

1. Install dependencies as above
2. Edit the `CONFIG` block at the top of `time_projection.py`
3. Run:
```bash
python time_projection.py
```

## Inputs

| Variable | Description |
|---|---|
| `sc_df_path` | Path to a parquet file with columns `ID`, `Frame`, `x`, `y` |
| `image_path` | Path to an OME-NGFF zarr — expects shape `(T, C, Z, Y, X)` under `images` |
| `mask_path` | Path to an OME-NGFF labels zarr — expects `labels/<name>/0` |
| `target_id` | The cell ID string to animate |
| `image_scale` | Pixel size (µm) used to convert tracked coordinates to pixel space |

## How it works

1. **Crop** — the cell bounding box is computed from the tracked `x`/`y` coordinates plus padding, and both the image stack and segmentation are sliced to that region.
2. **Focal mask** — the segmentation ID at the centre of the crop is sampled at each frame to isolate a single cell. The image stack is multiplied by this binary mask.
3. **Contours** — a smooth ring is extracted per frame via a Gaussian-softened morphological gradient: `dilate(blur(mask)) − blur(mask)`, yielding a float32 glow in `[0, 1]`.
4. **Layer architecture** — four napari layers build up the visual: a faint full-history **ghost**, a growing semi-transparent **history trail**, a bright **live cap** translated to the current Z position, and a **contour ring** that rides the cap.
5. **Milestone flashes** — at user-defined frames, a new contour layer is deposited at the current Z and its opacity decays toward a baseline over subsequent frames.
6. **Render** — `viewer.screenshot()` captures each state as a PNG. `imageio` with `libx264` stitches them into an MP4.

## Camera setup

Dial in camera angles interactively before running the full render.

**Notebook** — run cells 1–5, adjust the view in the napari window, then:
```python
viewer.camera.dict()
```
Paste `angles` and `zoom` back into the Configuration cell.

**Script** — uncomment the camera helper block near the bottom of `time_projection.py`, run the script once, adjust the view, print `viewer.camera.dict()`, then paste the values into the `CONFIG` block.
