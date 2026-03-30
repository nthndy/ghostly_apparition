"""
time_projection.py
------------------
Renders a single-cell time-projection animation using napari.

Each timepoint is stacked along the Z axis to build a 3D temporal tower,
screenshotted frame-by-frame, and compiled into an MP4.

Usage
-----
Edit the CONFIG block below, then run:
    python time_projection.py
"""

import os
import glob
import numpy as np
import pandas as pd
import zarr
import napari
import imageio.v2 as imageio
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter
from skimage.morphology import binary_dilation, disk


# =============================================================================
# CONFIG — edit these before running
# =============================================================================

SC_DF_PATH   = '/path/to/sc_df.parquet'
IMAGE_PATH   = '/path/to/acquisition/{acq_id}.zarr'   # OME-NGFF image zarr
MASK_PATH    = '/path/to/labels/{acq_id}.zarr'         # OME-NGFF labels zarr
OUTPUT_PATH  = 'time_projection.mp4'
TEMP_DIR     = '_tmp_frames'

TARGET_ID    = '330.3.5.PS0000'
IMAGE_SCALE  = 5.04    # pixel size (µm) used to scale tracked coordinates
CROP_PADDING = 250     # pixels of padding around the cell bounding box

Z_SCALE            = 10
CHANNEL_COLORMAPS  = ['green', 'magenta']
CHANNEL_CONTRAST   = [(0, 2600), (0, 900)]   # (min, max) per channel
MILESTONE_FRAMES   = list(range(0, 75, 5))
FLASH_DECAY_RATE   = 0.15
BASELINE_OPACITY   = 0.40
CAMERA_ANGLES      = (1.31, -3.08, -51.21)   # set after manual inspection (see below)
CAMERA_ZOOM        = 1.21
FPS                = 6
HOLD_FINAL_SECONDS = 3

# =============================================================================


def load_data(target_id, sc_df_path, image_path, mask_path):
    """Load single-cell dataframe, image stack, and segmentation."""
    df    = pd.read_parquet(sc_df_path)
    sc_df = df[df['ID'] == target_id].copy()

    acq_id = f"({int(target_id.split('.')[1])}, {int(target_id.split('.')[2])})"

    # OME-NGFF image — shape (T, C, Z, Y, X); max-project Z
    zarr_group      = zarr.open(image_path.format(acq_id=acq_id), mode='r')
    loaded_max_proj = zarr_group.images[...].max(axis=2)   # → (T, C, Y, X)

    # OME-NGFF labels — shape (T, Y, X) or (T, Z, Y, X)
    mask_store   = zarr.open(mask_path.format(acq_id=acq_id), mode='r')
    label_name   = list(mask_store['labels'])[0]
    segmentation = np.asarray(mask_store[f'labels/{label_name}/0'])
    if segmentation.ndim == 4:
        segmentation = segmentation.max(axis=1)   # → (T, Y, X)

    return sc_df, loaded_max_proj, segmentation


def crop_and_mask(sc_df, loaded_max_proj, segmentation, image_scale, crop_padding):
    """Crop to single-cell bounding box and build focal mask."""
    frames   = sc_df['Frame'].values.astype(int)
    x_coords = sc_df['y'].values * image_scale
    y_coords = sc_df['x'].values * image_scale

    x_min, x_max = int(x_coords.min() - crop_padding), int(x_coords.max() + crop_padding)
    y_min, y_max = int(y_coords.min() - crop_padding), int(y_coords.max() + crop_padding)

    sub_stack = loaded_max_proj[frames.min():frames.max()+1, :, x_min:x_max, y_min:y_max]
    sub_seg   = segmentation[frames.min():frames.max()+1, x_min:x_max, y_min:y_max]

    # Sample the seg ID at the crop centre to identify the focal cell
    mid_y, mid_x   = sub_seg.shape[1] // 2, sub_seg.shape[2] // 2
    target_seg_ids = sub_seg[:, mid_y, mid_x]

    focal_mask = np.zeros_like(sub_seg, dtype=np.float32)
    for t, sid in enumerate(target_seg_ids):
        if sid > 0:
            focal_mask[t] = (sub_seg[t] == sid).astype(np.float32)

    masked_stack = (sub_stack.astype(np.float32) * focal_mask[:, np.newaxis]).astype(sub_stack.dtype)

    return masked_stack, focal_mask


def build_contour_stack(focal_mask, sigma=1.5, dilation_radius=2):
    """
    Gaussian-softened morphological gradient contour.
    Returns float32 ring in [0, 1] per frame.
    """
    selem         = disk(dilation_radius)
    contour_stack = np.zeros_like(focal_mask, dtype=np.float32)

    for t in tqdm(range(focal_mask.shape[0]), desc='Building contours'):
        if not np.any(focal_mask[t]):
            continue
        soft             = gaussian_filter((focal_mask[t] > 0).astype(np.float64), sigma=sigma)
        contour_stack[t] = (binary_dilation(soft > 0.5, selem).astype(np.float32)
                            - (soft > 0.5).astype(np.float32))

    return contour_stack


def setup_viewer(masked_stack, contour_stack, z_scale, camera_zoom, camera_angles,
                 channel_colormaps):
    """Initialise the napari viewer and all layers."""
    viewer = napari.Viewer(title='Time Projection', ndisplay=3)

    viewer.add_image(masked_stack[:, 0], name='Ghost',
                     scale=(z_scale, 1, 1), colormap=channel_colormaps[0],
                     opacity=0.02, blending='additive')

    mphi_hist = viewer.add_image(masked_stack[0:1, 0], name='Mphi_Hist',
                                  scale=(z_scale, 1, 1), colormap=channel_colormaps[0],
                                  opacity=0.2, blending='additive')
    mtb_hist  = viewer.add_image(masked_stack[0:1, 1], name='Mtb_Hist',
                                  scale=(z_scale, 1, 1), colormap=channel_colormaps[1],
                                  opacity=0.2, blending='additive')

    mphi_top  = viewer.add_image(masked_stack[0:1, 0], name='Mphi_Top',
                                  colormap=channel_colormaps[0], blending='additive')
    mtb_top   = viewer.add_image(masked_stack[0:1, 1], name='Mtb_Top',
                                  colormap=channel_colormaps[1], blending='additive')

    live_cont = viewer.add_image(contour_stack[0:1], name='Live_Cont',
                                  colormap='cyan', blending='additive', opacity=0.99)
    live_cont.contrast_limits = (0, 1)

    viewer.camera.center = (0, masked_stack.shape[2] / 2, masked_stack.shape[3] / 2)
    viewer.camera.zoom   = camera_zoom
    viewer.camera.angles = camera_angles

    return viewer, mphi_hist, mtb_hist, mphi_top, mtb_top, live_cont


def render(viewer, masked_stack, contour_stack, mphi_hist, mtb_hist, mphi_top, mtb_top,
           live_cont, z_scale, channel_contrast, milestone_frames, flash_decay_rate,
           baseline_opacity, temp_dir):
    """Frame-by-frame reveal loop."""
    os.makedirs(temp_dir, exist_ok=True)
    milestone_set = set(milestone_frames)
    T = masked_stack.shape[0]

    for i in tqdm(range(T), desc='Rendering'):

        if i > 0:
            mphi_hist.data, mtb_hist.data = masked_stack[:i, 0], masked_stack[:i, 1]

        mphi_top.data  = masked_stack[i:i+1, 0]
        mtb_top.data   = masked_stack[i:i+1, 1]
        live_cont.data = contour_stack[i:i+1]

        live_cont.contrast_limits = (0, 1)
        mphi_top.contrast_limits  = channel_contrast[0]
        mtb_top.contrast_limits   = channel_contrast[1]
        mphi_hist.contrast_limits = channel_contrast[0]
        mtb_hist.contrast_limits  = channel_contrast[1]

        z = i * z_scale
        mphi_top.translate = mtb_top.translate = live_cont.translate = (z, 0, 0)

        if i in milestone_set:
            viewer.add_image(contour_stack[i:i+1], name=f'Mile_{i}',
                             colormap='cyan', opacity=1.0,
                             translate=(z, 0, 0), blending='additive')

        for lyr in viewer.layers:
            if lyr.name.startswith('Mile_') and lyr.opacity > baseline_opacity:
                lyr.opacity = max(baseline_opacity, lyr.opacity - flash_decay_rate)

        viewer.camera.center = (z, viewer.camera.center[1], viewer.camera.center[2])

        for lyr in viewer.layers:
            lyr.refresh()
        viewer.screenshot(path=os.path.join(temp_dir, f'frame_{i:04d}.png'))


def compile_mp4(temp_dir, output_path, fps, hold_final_seconds):
    """Stitch PNGs into an MP4 and clean up."""
    pngs = sorted(glob.glob(os.path.join(temp_dir, 'frame_*.png')))

    with imageio.get_writer(output_path, fps=fps, codec='libx264', quality=9) as w:
        for f in pngs:
            w.append_data(imageio.imread(f))
        final = imageio.imread(pngs[-1])
        for _ in range(fps * hold_final_seconds):
            w.append_data(final)

    for f in pngs:
        os.remove(f)

    print(f'Saved → {output_path}')


# =============================================================================
# Camera helper — run this separately to dial in your view
# =============================================================================
# viewer = napari.Viewer(ndisplay=3)
# viewer.add_image(masked_stack, channel_axis=1, colormap=['green','magenta'],
#                  blending='additive', scale=(Z_SCALE, 1, 1))
# # Adjust the view manually, then:
# print(viewer.camera.dict())
# =============================================================================


if __name__ == '__main__':
    print(f'Loading data for {TARGET_ID}...')
    sc_df, loaded_max_proj, segmentation = load_data(
        TARGET_ID, SC_DF_PATH, IMAGE_PATH, MASK_PATH)

    print('Cropping and masking...')
    masked_stack, focal_mask = crop_and_mask(
        sc_df, loaded_max_proj, segmentation, IMAGE_SCALE, CROP_PADDING)
    print(f'  Stack shape: {masked_stack.shape}  (T, C, Y, X)')

    contour_stack = build_contour_stack(focal_mask)

    viewer, mphi_hist, mtb_hist, mphi_top, mtb_top, live_cont = setup_viewer(
        masked_stack, contour_stack, Z_SCALE, CAMERA_ZOOM, CAMERA_ANGLES,
        CHANNEL_COLORMAPS)

    render(viewer, masked_stack, contour_stack,
           mphi_hist, mtb_hist, mphi_top, mtb_top, live_cont,
           Z_SCALE, CHANNEL_CONTRAST, MILESTONE_FRAMES,
           FLASH_DECAY_RATE, BASELINE_OPACITY, TEMP_DIR)

    compile_mp4(TEMP_DIR, OUTPUT_PATH, FPS, HOLD_FINAL_SECONDS)
    viewer.close()
