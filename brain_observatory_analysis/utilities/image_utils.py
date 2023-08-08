import numpy as np
import pandas as pd
import os
import colorsys
import imageio
from pathlib import Path
from typing import Union
import brain_observatory_qc.data_access.from_lims as from_lims
from tifffile import TiffWriter

########################################################################
# Image outputs
########################################################################


def save_gif(image_stack: np.ndarray,
             gif_folder_path: Union[str, Path],
             fn: str,
             clip_image: bool = True,
             vmax_percentile: float = 99,
             frame_duration: float = 0.1) -> None:
    """
    Save a 3D image stack as an animated gif.

    Parameters
    ----------
    image_stack : np.ndarray
        3D image stack to save as animated gif. Must be 3D. tyx format.
    gif_folder_path : Union[str, Path]
        Path to folder where gif will be saved.
    fn : str
        Filename of gif.
    clip_image : bool, optional
        If True, clip image to vmax_percentile. The default is True.
    vmax_percentile : float, optional
        Percentile to clip image to. The default is 99.
    frame_duration : float, optional
        Duration of each frame in seconds. The default is 0.1.

    Returns
    -------
    None.

    """

    if not isinstance(image_stack, np.ndarray):
        raise TypeError('image_stack must be a numpy array')

    if len(image_stack.shape) != 3:
        raise ValueError('image_stack must be 3D')

    if not os.path.exists(gif_folder_path):
        os.makedirs(gif_folder_path)

    images = []
    for img in image_stack:

        # set max of img to 99th percentile
        if clip_image:
            vmax = np.percentile(img, vmax_percentile)
            img = img.clip(0, vmax)

        # convert to 8-bit
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = (img * 255).astype(np.uint8)

        images.append(img)

    # add "gif" to end of filename if not already there
    if fn[-4:] != '.gif':
        fn = fn + '.gif'

    gif_path = Path(gif_folder_path + fn)
    imageio.mimsave(gif_path, images, duration=frame_duration)

    # print message
    print(f"Saved gif to {gif_path} with {image_stack.shape[0]} frames")


def save_tiff(images, filename):
    """Save images as tiffs

    Parameters
    ----------
    images : list of numpy.ndarray
            List of images to save as tiffs
    filename : str
            Filename to save tiffs as
    # TODO: make more robust
    Returns
    -------
    None
    """
    with TiffWriter(filename) as tif:
        for image in images:
            tif.save(image)

########################################################################
# from_lims
########################################################################


def get_motion_correction_crop_xy_range(oeid):
    """Get x-y ranges to crop motion-correction frame rolling

    Note: this gets the range from the ophsy_etl_pipeline output

    Parameters
    ----------
    oeid : int
        ophys experiment ID

    Returns
    -------
    list, list
        Lists of y range and x range, [start, end] pixel index
    """
    # TODO: validate in case where max < 0 or min > 0 (if there exists an example)
    motion_df = pd.read_csv(from_lims.get_motion_xy_offset_filepath(oeid))
    max_y = np.ceil(max(motion_df.y.max(), 1)).astype(int)
    min_y = np.floor(min(motion_df.y.min(), 0)).astype(int)
    max_x = np.ceil(max(motion_df.x.max(), 1)).astype(int)
    min_x = np.floor(min(motion_df.x.min(), 0)).astype(int)
    range_y = [-min_y, -max_y]
    range_x = [-min_x, -max_x]

    return range_y, range_x


def generate_distinct_colors(num_colors):
    """Generate distinct colors
    
    Parameters
    ----------
    num_colors : int
        Number of colors to generate
        
    Returns
    -------
    list 
        List of RGB tuples
    """

    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        rgb = colorsys.hsv_to_rgb(hue, 1, 1)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors
