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

"""Module for calculation of PSF."""
from pathlib import Path
import logging

import numpy as np
import click
from scipy import optimize
from scipy.ndimage.filters import gaussian_filter

from insilico_vsdi.depth_point_spread.ray_trace import trace
from insilico_vsdi.utils import read_json

logger = logging.getLogger(__name__)


class Camera:
    """Camera for tracing of photons."""

    def __init__(self, f1, f2, f1_no, f2_no, sensor_dims, image_dims):
        """Create a camera.

        Args:
            f1 (float): objective focal length [microns]
            f2 (float): tube focal length [microns]
            f1_no (float): f-number of objective
            f2_no (float): f-number of tube
            sensor_dims (tuple(int,int)): sensor dimensions
            image_dims (tuple(int,int)): image plane resolution
        """
        self.f1 = f1
        self.f2 = f2
        self.f1_no = f1_no
        self.f2_no = f2_no
        self.sensor_dims = sensor_dims
        self.image_dims = image_dims

    @property
    def d1(self):
        """Diameter of objective lens [microns]."""
        return self.f1 / self.f1_no

    @property
    def d2(self):
        """Diameter of tube lens [microns]."""
        return self.f2 / self.f2_no

    @property
    def na1(self):
        """Numerical aperture of objective (image space)."""
        return 1. / (2 * self.f1_no)

    @property
    def p(self):
        # pylint: disable=line-too-long
        """Infinity space path length.

        www.edmundoptics.com/resources/application-notes/microscopy/using-tube-lenses-with-infinity-corrected-objectives/
        """
        return (self.d2 - self.d1) * self.f2 / (2 * self.f1 * self.na1)


def _gaussian(height, x0, y0, sigx, sigy):
    """Two-dimensional Gaussian surface function for fitting to image plane data.

    Args:
        height (float): height of surface at peak
        x0 (float): x coordinate of peak
        y0 (float): y coordinate of peak
        sigx (float): Gaussian width (standard deviation) in x direction
        sigy (float): Gaussian width (standard deviation) in y direction

    Returns:
        function: Gaussian function g(x,y) (of x and y) describing surface
    """
    sigx, sigy = float(sigx), float(sigy)
    return lambda x, y: height * np.exp(-(((x0 - x) / sigx) ** 2 + ((y0 - y) / sigy) ** 2) / 2)


def _moments(detector_img):
    """Extracts Gaussian moments from a detector plane image.

    Args:
        detector_img (numpy.ndarray): detector plane image

    Returns:
        numpy.ndarray: Gaussian moments as numpy array of 5 items:
            float: height of surface at peak
            float: x coordinate of peak
            float: y coordinate of peak
            float: Gaussian width (standard deviation) in x direction
            float: Gaussian width (standard deviation) in y direction
    """
    total = detector_img.sum()
    X, Y = np.indices(detector_img.shape)
    x = (X * detector_img).sum() / total
    y = (Y * detector_img).sum() / total
    col = detector_img[:, int(y)]
    sigx = np.sqrt(np.abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
    row = detector_img[int(x), :]
    sigy = np.sqrt(np.abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
    height = detector_img.max()
    return np.array([height, x, y, sigx, sigy])


def _fit_gaussian(detector_img):
    """Chooses Gaussian moments of the best fit Gaussian surface for image plane data.

    Choice is made by least squares procedure.

    Args:
        detector_img (numpy.ndarray): detector plane image

    Returns:
        numpy.ndarray: Gaussian moments of the best fit gaussian
    """

    def _estimate_moments(moments):
        gaussian_fit = _gaussian(*moments)(x=img_indices[0], y=img_indices[1])
        return np.ravel(gaussian_fit - detector_img)

    img_indices = np.indices(detector_img.shape)
    initial_moments = _moments(detector_img)
    best_moments, _ = optimize.leastsq(_estimate_moments, initial_moments)
    return best_moments


def _create_camera(config_path):
    """Creates a camera instance for tracing of photons.

    Args:
        config_path (str): filepath to a VSDI config

    Returns:
        Camera: camera instance
    """
    config = read_json(config_path)
    sensor_dim = config["emsim-vsd-args"]["sensor-dim"]
    sensor_res = config["emsim-vsd-args"]["sensor-res"]
    return Camera(f1=50e3, f2=135e3, f1_no=0.95, f2_no=2.0,
                  sensor_dims=(sensor_dim, sensor_dim), image_dims=(sensor_res, sensor_res))


def calculate_and_save(um_px, kernel_dir, config_path, output):
    """Calculates and saves PSF.

    PSF is represented as an array of sigmas (std) of gaussian kernels of photons propagation
    along the depth of cortical volume. You can use this file later for the main pipeline in
    config["movie-args"]["point-spread-blur"].

    Args:
        um_px (float): microns per voxel side length
        kernel_dir (str): path to a folder where Monte-Carlo kernel files are stored. Those files
            represent photon spread on image plane for a point source of light after passage through
            tissue and microscope. Each file corresponds to a different depth for the point source
            along the y-axis of the circuit.
        config_path (str): filepath to a VSDI config
        output (str): path to a file where to store PSF sigmas. '.npy' extension will be appended to
            the file name if it does not already have one. Also an additional file will be stored
            next to it with the same name plus a suffix '_raw' for debug purposes.
    """
    # pylint: disable=too-many-locals
    fns = list(Path(kernel_dir).glob('*sensor.photons'))
    if len(fns) == 0:
        raise ValueError('no kernel files found')

    camera = _create_camera(config_path)
    sigmas = np.zeros((len(fns), 2))
    logger.info('no pre-computed sigmas found... calculating now')
    for fn in fns:
        name_parts = fn.name.split('_')
        idx = int(name_parts[0])
        # find y-value (depth) of this photon point source
        depth = float(name_parts[1].lstrip('depth')) * 1000  # in microns (not mm)
        detector_img, _, _ = trace(fn, camera, False)
        # spatially filter and normalize image plane data for improved fitting
        detector_img = gaussian_filter(detector_img, 1)
        detector_img /= detector_img.max()

        _, _, _, sigx, sigy = _fit_gaussian(detector_img)

        # extract standard deviation from Gaussian surface of best fit
        sigma = np.sqrt(sigx * sigy) * um_px
        sigmas[idx, :] = [depth, sigma]
        logger.debug('calculated sigma: %s', sigma)

        # fit = _gaussian(height, x, y, sigx, sigy)
        # plt.imshow(img, cmap=plt.cm.viridis, origin='bottom')
        # plt.contour(fit(*np.indices(img.shape)), colors='w')
        # plt.show(block=True)

    # fit a curve to the empirical vector of (depth-dependent) Gaussian widths using a double
    # exponential function
    p0 = (12, 4, -0.001, 0.0001, 10)
    f = lambda x, A, B, u, v, C: A * np.exp(u * x) + B * np.exp(v * x) + C
    # pylint: disable=unbalanced-tuple-unpacking
    popt, _ = optimize.curve_fit(f, sigmas[:, 0], sigmas[:, 1], p0=p0)
    depths = np.linspace(sigmas[0, 0], sigmas[-1, 0], int(sigmas[-1, 0]) + 1)
    # save fitted depth-dependent curve of Gaussian widths (i.e. "point spread")
    sigmas_fit = f(depths, *popt)
    output = Path(output)
    # put '0' at cortical surface
    np.save(output.parent / (output.stem + '_raw'), np.array([depths, sigmas[:, 1]]).T)
    # put '0' at cortical surface
    np.save(output, np.array([depths, sigmas_fit]).T)


@click.command(help=calculate_and_save.__doc__)
@click.option('--um_px', default=10.0283, show_default=True)
@click.option('--kernel_dir', type=click.Path(exists=True, file_okay=False),
              help='path to a folder where Monte-Carlo kernel files are stored')
@click.option('--config', help='filepath to a VSDI config', required=True)
@click.option('--output', type=click.Path(), help='path to a file where to store PSF sigmas')
def cmd(um_px, kernel_dir, config, output):
    """Cli for `calculate_and_save` function."""
    calculate_and_save(um_px, kernel_dir, config, output)
