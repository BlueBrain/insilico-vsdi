# Copyright (C) 2017-2020 Blue Brain Project / EPFL
#
# This file is part of Blue Brain InsilicoVSDI library <https://github.com/BlueBrain/insilico-vsdi>
#
# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License version 3.0 as published
# by the Free Software Foundation.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""Module that generates image frames from volumetric voltage data."""
import logging
from pathlib import Path

import click
import numpy as np
from scipy.ndimage import gaussian_filter
from mhd_utils import load_raw_data_with_mhd

from insilico_vsdi.utils import read_json, find_nearest

logger = logging.getLogger(__name__)


def _accumulate_img(volume, sigmas, um_px, isurf=0):
    """Accumulates horizontal (x-z plane) slices of volume data into an image array.

    Convolves each slice with a depth-dependent Gaussian spatial filter.
    Args:
        volume (numpy.ndarray): 3D array of voxel data
        sigmas (numpy.ndarray): 1D array of depth-dependent standard deviations for use in Gaussian
            kernel
        um_px (float): microns per image pixel
        isurf (int): depth index (in data volume) at which surface begins

    Returns:
    numpy.ndarray: 2D image array of raw (un-normalized) VSDI pixels
    numpy.ndarray: 2D image array of raw (un-normalized) VSDI pixels, but no convolution with
        Gaussian kernel (for debugging purposes)
    """
    dims = np.shape(volume)[1:]
    image = np.zeros(dims)
    debug = np.zeros(dims)
    for k, slice_ in enumerate(volume):
        if k < isurf:
            idx = 0
        else:
            depth = (k - isurf) * um_px
            _, idx = find_nearest(sigmas[:, 0], depth)
        blurred = gaussian_filter(slice_, sigmas[idx, 1] / um_px)
        image += blurred
        debug += slice_
    return image, debug


def gen_frames(frames_output, config, skip_existing=True):
    """Generates raw VSDI images by summating along the y-axis (depth axis) of volumetric data.

    Data slice at each depth is convolved with a Gaussian spatial filter to correct for
    point-spread-related blur of emitted light (see *depth_point_spread*).

    Args:
        frames_output (str): frames output filepath
        config (dict): parsed VSDI config
        skip_existing (bool): skips processing for output files that already exist
    """
    logger.debug('Generating image frames for %s', frames_output)
    output_dirpath = Path(frames_output).parent
    sigmas = np.load(config['movie-args']['point-spread-blur'])
    volume_filepaths = list(output_dirpath.glob('*volume*.mhd'))
    if len(volume_filepaths) == 0:
        logging.warning('No volume voltage data is found at %s', output_dirpath)

    for volume_filepath in volume_filepaths:
        img_filepath = volume_filepath.parent / (volume_filepath.stem + '.npy')
        img_dbg_filepath = img_filepath.parent / (img_filepath.stem + '.dbg.npy')
        if skip_existing and img_dbg_filepath.exists():
            logger.debug('skipping existing image: %s', img_filepath.name)
            continue
        logger.debug('computing image: %s', img_filepath.name)
        volume, meta = load_raw_data_with_mhd(volume_filepath)
        um_px = meta['ElementSpacing'][0]
        volume = np.swapaxes(volume, 0, 1)  # put y-axis (depth) first
        volume = np.array([slice.T for slice in volume])  # transpose each slice for image indexing
        img, db = _accumulate_img(volume, sigmas, um_px)
        np.save(img_filepath, img)
        np.save(img_dbg_filepath, db)


@click.command(help=gen_frames.__doc__)
@click.option('--frames-output', help='frames output filepath', required=True)
@click.option('--config', help='filepath to a VSDI config', required=True)
@click.option('--skip-existing', help='skips processing for output files that already exist',
              is_flag=True)
def cmd(frames_output, config, skip_existing=True):
    """Cli for `gen_frames` function."""
    gen_frames(frames_output, read_json(config), skip_existing)
