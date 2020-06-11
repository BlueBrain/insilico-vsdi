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

"""Module that generates VSD movies from image frames."""
import re
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib
import numpy as np
import click
from natsort import natsorted

from insilico_vsdi.utils import read_json, find_nearest

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


def _timevec(frame_filenames):
    """Extracts time steps(in milliseconds) from image frame filenames.

    Args:
        frame_filenames (List[str]): list of image frame filenames

    Returns:
        numpy.ndarray: times, float array of simulation time steps
    """
    str_times = [re.findall(r"\d+.\d+", f) for f in frame_filenames]
    for i, str_time in enumerate(str_times):
        assert len(str_time) == 1 and float(str_time[0]) > 0, \
            'Incorrect time steps in frame filename {}'.format(frame_filenames[i])
    return np.array([float(str_time[0]) for str_time in str_times])


def _norm_frames(frames, izstart, izfinish, norm_special):
    """Normalizes image frames.

    According to: (F-F0)/F0 (percent change over baseline), where F is an image frame, and F0 is an
    average of baseline frames.

    Args:
        frames (numpy.ndarray): image stack of un-normalized VSD frames
        izstart (int): index of starting frame to calculate normalizing (baseline) image frame
        izfinish (int): index of ending frame to calculate normalizing (baseline) image frame
        norm_special (string): path to a file with a custom norm for frames.

    Returns:
        numpy.ndarray: normalized stack of image frames
        None|numpy.ndarray: fractional baseline frame, used for calculating the fractional
        contribution of a subpopulation of neurons to the overall VSD signal
    """
    # exclude zero frames from normalization
    frames[frames == 0] = np.nan
    if norm_special is not None:
        data = np.load(norm_special)
        data[data == 0] = np.nan
        zframe = np.nanmean(data[izstart: izfinish + 1, :, :], axis=0)
        zframe_alt = np.nanmean(frames[izstart: izfinish + 1, :, :], axis=0)
        frac = zframe_alt / zframe
    else:
        zframe = np.nanmean(frames[izstart: izfinish + 1, :, :], axis=0)
        frac = None
    normed_frames = frames - np.repeat(zframe[np.newaxis, :, :], frames.shape[0], axis=0)
    normed_frames /= zframe
    # normed_frames = np.asarray([normed_frames[k, :, :] for k in range(normed_frames.shape[0])])
    return normed_frames, frac


def _make_mask(soma_pixels, res):
    """Calculates mask matrix for VSD movies.

    Mask preserves all image pixels underneath which a soma is present.  Excludes all pixels that
    don't contain somata (as viewed from the top of the column downward along the y-axis).

    Args:
        soma_pixels (numpy.ndarray): array with rows: (gid, NaN, x, y, z, NaN, i, j, k)
        res (int): sensor resolution (number of pixels in rows/columns; assumes square geometry)

    Returns:
        numpy.ndarray: mask matrix containing 1s where soma is present and NaNs otherwise
    """
    # clip data to area defined by soma locations
    nrows = res
    ncols = res
    mask = np.zeros((nrows, ncols))

    # build mask from pixels containing somas
    for row in range(soma_pixels.shape[0]):
        i, _, k = soma_pixels[row][6:]
        i = int(nrows - 1 - i)
        k = int(k)
        if (i < res - 1) and (k < res - 1):
            mask[i, k] = 1

    mask[mask == 0] = np.nan
    return mask


def _clip_data(data, mask, indices=None):
    """Clips VSD movie data to specified mask matrix.

    Mask typically contains 1s for pixels underneath which a soma is present, and NaNs otherwise.

    Args:
        data (numpy.ndarray): VSD movie data
        mask (numpy.ndarray): mask matrix
        indices (tuple of two 1-D vectors): if not None, each vector specifies particular indices of
        elements in data for which to apply masking operation

    Returns:
        numpy.ndarray: data after application of mask matrix
    """
    if indices is not None:
        ii, kk = np.meshgrid(indices[0], indices[1], indexing="ij")
        data[:, ii, kk] = data[:, ii, kk] * np.repeat(mask[np.newaxis, :, :], data.shape[0], axis=0)
        return data
    else:
        data[data == 0] = np.nan
        return data * np.repeat(mask[np.newaxis, :, :], data.shape[0], axis=0)


def _animate_frames(frames, vlims=None):
    """Animation tool for VSD movies.

    Args:
        frames (numpy.ndarray): image stack of normalized VSD frames
        vlims (tuple of two floats, optional): defines limits for plot colors

    Returns:
        animation object: VSD movie
    """

    def updateFig(n):
        im.set_array(frames[n])
        return (im,)

    fig = plt.figure()
    cmap = plt.get_cmap("jet")
    cmap.set_bad(color="k", alpha=1.0)
    im = plt.imshow(frames[0], cmap=cmap)

    # create animation
    if vlims is not None:
        plt.clim(vlims[0], vlims[1])
    movie = animation.FuncAnimation(fig, updateFig, frames=len(frames), interval=10, blit=True)
    return movie


def make_movie(frames_output, movie_output, config, skip_existing=True):
    """Generates a VSDI movie.

    The stack of images is stitched together and normalized by the average of the first several
    frames to produce a VSDI movie.

    Args:
        frames_output (string): frames output filepath
        movie_output (string): filepath specifying where VSD movie should be stored
        config (dict): parsed VSDI config
        skip_existing (bool): skips processing for output files that exist already
    """
    # pylint: disable=too-many-locals
    movie_filepath = Path(movie_output)
    assert movie_filepath.name == movie_filepath.stem, \
        'movie_output must be a path without file extension'
    movie_dirpath, movie_name = movie_filepath.parent, movie_filepath.stem
    if skip_existing and movie_dirpath.joinpath(movie_name + '.mp4').exists():
        logger.debug("Movie %s exists already, skipping", movie_filepath)
        return
    movie_args = config["movie-args"]
    logger.debug("Generating movie for %s with args: %s", movie_filepath, movie_args)
    frames_dirpath = Path(frames_output).parent
    frame_filepaths = set(frames_dirpath.glob("*.npy")) - set(frames_dirpath.glob("*.dbg.npy"))
    frame_filenames = natsorted([filepath.name for filepath in frame_filepaths])

    # load json file parameters
    times = _timevec(frame_filenames)
    sensor_res = config["emsim-vsd-args"]["sensor-res"]
    soma_filepath = next(frames_dirpath.glob("*soma_pixels*"))
    soma_pixels = np.genfromtxt(soma_filepath)
    _, izstart = find_nearest(times, movie_args["zstart"])
    _, izfinish = find_nearest(times, movie_args["zfinish"])
    norm_special = movie_args["norm-special"]

    frames = []
    logger.debug("loading frames %s", frame_filenames)
    for frame_filename in frame_filenames:
        frames.append(np.load(frames_dirpath / frame_filename))
    frames = np.asarray(frames)
    np.save(movie_dirpath / (movie_name + "_raw"), frames)

    # generate movie data
    frames, zfrac = _norm_frames(frames, izstart, izfinish, norm_special)
    mask = _make_mask(soma_pixels, sensor_res)
    frames = _clip_data(frames, mask)
    np.save(movie_filepath, frames)
    np.save(movie_dirpath / (movie_name + "_frac"), zfrac)
    movie = _animate_frames(frames, [np.nanmin(frames) / 2, np.nanmax(frames) / 2])

    # setup animation writer
    # Writer = animation.writers['avconv']
    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=200, bitrate=3000)
    movie.save(movie_dirpath / (movie_name + ".mp4"), writer=writer)


@click.command(help=make_movie.__doc__)
@click.option('--frames-output', help='frames output filepath', required=True)
@click.option('--movie-output', help='movie output filepath', required=True)
@click.option('--config', help='filepath to a VSDI config', required=True)
@click.option('--skip-existing', help='skips processing for output files that already exist',
              is_flag=True)
def cmd(frames_output, movie_output, config, skip_existing=True):
    """Cli for `make_movie` function."""
    make_movie(frames_output, movie_output, read_json(config), skip_existing)
