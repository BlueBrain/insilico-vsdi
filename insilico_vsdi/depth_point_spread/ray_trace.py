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

"""Module for detecting photons propagation through the optical system."""
import enum
import logging
import numpy as np

from matplotlib.cm import ScalarMappable, rainbow  # pylint: disable=no-name-in-module
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class LensType(enum.Enum):
    """Type of lens."""
    objective = 1
    tube = 2


def _to_polar_coordinates(photons):
    """Extracts polar coordinates for each photon from cartesian coordinates.

    Args:
        photons (numpy.ndarray): Monte Carlo photon data. An array with x and y positions of photons
            after they scatter through and exit cortical tissue, impinging on a plane that bisects
            the objective lens.

    Returns:
        numpy.ndarray: polar angles, angles with respect to plane perpendicular to line passing
            through center of optical system
        numpy.ndarray: polar radii of photons, distance from line passing through center of optical
            system
    """
    x = photons[:, 0]
    y = photons[:, 1]
    phi = np.arctan2(y, x)
    r = np.sqrt(x ** 2 + y ** 2)
    return phi, r


def _photons2pixels(r, phi, dims):
    """Converts detected photons to images.

    Photons impinging on detector(in polar coordinates) are converted into an image matrix.

    Args:
        r (numpy.ndarray): photons polar radii
        phi (numpy.ndarray): photons polar angles
        dims (tuple): sensor dimensions

    Returns:
        numpy.ndarray: array of detector (image) pixels
        numpy.ndarray: x-dimension edges in image array
        numpy.ndarray: y-dimension edges in image array
    """
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    arr, xe, ye = np.histogram2d(x, y, bins=[dims[0], dims[1]])
    return arr, xe, ye


def _exclude_missed(lens, r, camera, *args):
    """Excludes missed photons.

    Photons are missed if they scatter at an angle sufficiently steep such that they miss the one of
    the lenses altogether and exit the optical system.

    Args:
        lens (LensType): type of lens
        r (numpy.ndarray): polar radii of photons
        camera (Camera): used camera
        args (positional args): additional data structures to which restriction is applied

    Returns:
        numpy.ndarray: radii of non-excluded photons
        tuple: data structures with elements corresponding to only non-excluded photons
        numpy.ndarray: indices of non-excluded photons
    """
    assert lens in (LensType.objective, LensType.tube,), 'Must be a known lens type'

    if lens is LensType.objective:
        ind = np.where(np.abs(r) < camera.d1 / 2.0)[0]
    else:
        ind = np.where(np.abs(r) < camera.d2 / 2.0)[0]

    return r[ind], tuple(arg[ind] for arg in args), ind


def _apply_transform(r, theta, phi, camera):
    """Applies a ray transfer matrix transform to the Monte Carlo photon data.

    The ray transfer propagates each photon through the tandem-lens optical system accounting for
    refraction through each lens.
    System is:
        IN: L1 --> (d) --> L2 --> (F2) --> :OUT
    IN is at the surface of a first thin lens, L1. OUT is an image. Plane at the focus of the
    second thin lens, L2. The two lenses are separated by a distance P.

    Args:
        r (numpy.ndarray): photons polar radii
        theta (numpy.ndarray): photons incidence angles, angles with respect to line passing through
            center of optical system
        phi (numpy.ndarray): photons polar angles
        camera (Camera): used camera

    Returns:
        tuple: ray transfer matrices representing propagation through optical system
            numpy.ndarray: matrix representing refraction through objective lens
            numpy.ndarray: matrix representing passage through optical system until reaching tube
                lens
            numpy.ndarray: matrix representing passage through optical system until refraction
                through tube lens
            numpy.ndarray: matrix representing passage through optical system until reaching
                detector plane
        numpy.ndarray: polar radii of photons on detector plane
        numpy.ndarray: incidence angles of photons on detector plane
        numpy.ndarray: polar angles of photons on detector plane
        tuple: indices of exclude photons (those that miss the objective lens and/or tube lens)
            numpy.ndarray: indices of photons to be excluded due to missing objective lens
            numpy.ndarray: indices of photons to be excluded due to missing tube lens
    """
    # L1: refraction through the objective lens
    L1 = np.array([[1, 0], [-1 / camera.f1, 1]])
    # L2: refraction through the tube lens
    L2 = np.array([[1, 0], [-1 / camera.f2, 1]])
    # T1: translation from the objective lens to the tube lens
    T1 = np.array([[1, camera.p], [0, 1]])
    # T2: translation from the tube lens to the detector plane
    T2 = np.array([[1, camera.f2], [0, 1]])

    A = L1
    B = np.linalg.multi_dot([T1, L1])
    C = np.linalg.multi_dot([L2, T1, L1])
    M = np.linalg.multi_dot([T2, L2, T1, L1])

    r, (theta, phi), ind0 = _exclude_missed(LensType.objective, r, camera, theta, phi)
    v = B.dot([r, theta])
    r, (theta, phi), ind1 = _exclude_missed(LensType.tube, v[0, :], camera, v[1, :], phi)
    v = np.linalg.multi_dot([T2, L2]).dot([r, theta])

    # (A, B, C, M), r, theta, phi, (ind0, ind1)
    return (A, B, C, M), v[0, :], v[1, :], phi, (ind0, ind0[ind1])


def _correct_angles(r, phi):
    """Corrects negative radii introduced after matrices transformation.

    Flips the polar angle and makes the radius positive again.
    """
    iflip = np.where(r < 0)[0]
    to_flip = phi[iflip]
    ineg = iflip[to_flip < 0]
    ipos = iflip[to_flip >= 0]
    phi[ineg] += np.pi
    phi[ipos] -= np.pi
    r[iflip] *= -1

    return r, phi


def _convert_photons(photons, camera):
    """Converts raw Monte-Carlo photon data matrix to usable format.

    Raw input is has columns of form: (x, y, theta), where x and y are cartesian coordinates, and
    theta is the incidence angle of each photon.  Specifically: x and y coordinates are centered
    (so 0,0 is at center of lens) and converted from pixel units to physical units (microns).
    Theta is converted from degrees to radians.

    Args:
        photons (numpy.ndarray): Monte Carlo photon data. An array with x and y positions of photons
            after they scatter through and exit cortical tissue, impinging on a plane that bisects
            the objective lens.
        camera (Camera): used camera

    Returns:
        numpy.ndarray: transformed photon data array
    """
    photons[:, 0] = photons[:, 0] - (camera.sensor_dims[0] - 1) / 2.0
    photons[:, 1] = photons[:, 1] - (camera.sensor_dims[1] - 1) / 2.0
    photons[:, 0] = photons[:, 0] * camera.d1 / camera.sensor_dims[0]
    photons[:, 1] = photons[:, 1] * camera.d1 / camera.sensor_dims[1]
    photons[:, 2] = np.deg2rad(photons[:, 2])
    return photons


def _plot_rays(r0, theta0, phi0, T, ind, camera, color_map, plot="2d"):
    """Visualizes diffraction of light rays through two lenses (side view).

    Args:
        r0 (numpy.ndarray): initial photons polar radii
        theta0 (numpy.ndarray): initial photons incidence angles
        phi0 (numpy.ndarray): initial photons polar angles
        T (tuple): tuple of ray transfer matrices
        ind (tuple): tuple of photons indices to exclude
        camera (Camera): used camera
        color_map (Colormap): matplotlib colormap
        plot (string, optional): indicates whether plot should be visualized in 2D or 3D
    """
    # pylint: disable=too-many-locals
    DOWNSAMPLE = 5000  # factor by which to downsample rays for plotting
    OFFSET = 35e3  # distance [microns] between tissue surface and objective lens
    radii = np.zeros((len(r0), 5))

    radii[:, 0] = r0 - OFFSET * theta0

    radii[:, 1] = r0
    r1, theta1 = T[0].dot([r0[ind[0]], theta0[ind[0]]])

    radii[:, 2] = r0 + camera.p * theta0
    radii[ind[0], 2] = r1 + camera.p * theta1
    r2, theta2 = T[2].dot([r0[ind[1]], theta0[ind[1]]])

    radii[:, 3] = r0 + (camera.p + camera.f2) * theta0
    radii[ind[0], 3] = r1 + (camera.p + camera.f2) * theta1
    radii[ind[1], 3] = r2 + camera.f2 * theta2

    radii[:, 4] = np.ones((len(r0),)) * -1
    radii[ind[0], 4] = np.ones((len(ind[0]),)) * 0
    radii[ind[1], 4] = np.ones((len(ind[1]),)) * 1

    X = [-OFFSET, 0, camera.p, camera.p + camera.f2]
    cNorm = Normalize(vmin=0, vmax=np.abs(radii[:, 1] * np.sin(phi0)).max())
    scalarMap = ScalarMappable(norm=cNorm, cmap=color_map)

    lens0 = Ellipse(xy=np.array([X[1], 0]).T, width=1500, height=camera.d1, angle=0)
    lens1 = Ellipse(xy=np.array([X[2], 0]).T, width=1500, height=camera.d2, angle=0)

    assert plot in ('2d', '3d',), "please specify a valid plotting style ('2d' or '3d')"

    if plot == "2d":
        fig, ax = plt.subplots()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim([-OFFSET - 3000, camera.p + camera.f2 + 3000])
    ylim = camera.d1 * 2  # y-dimension for plotting
    ax.set_ylim([-ylim, ylim])

    ax.add_artist(lens0)
    ax.add_artist(lens1)
    lens0.set_alpha(0.5)
    lens1.set_alpha(0.5)

    for k, rad in enumerate(radii[::DOWNSAMPLE]):
        # rad[4] == -1, missed both lenses
        # rad[4] == 0, passed the first lens, but missed the second
        if rad[4] != 1:
            continue
        # made it through both lenses
        color = scalarMap.to_rgba(np.abs(rad[1] * np.sin(phi0[k * DOWNSAMPLE])))
        if plot == "2d":
            ax.plot(X, rad[:-1] * np.sin(phi0[k * DOWNSAMPLE]), c=color, alpha=0.1)
        else:
            ax.plot(
                X,
                rad[:-1] * np.sin(phi0[k * DOWNSAMPLE]),
                rad[:-1] * np.cos(phi0[k * DOWNSAMPLE]),
                c=color,
                alpha=0.1,
            )


def _plot_photons(r, phi, r0, phi0, theta0, T, ind, camera):
    # pylint: disable=too-many-locals
    """Plots how photons propagate through two lenses (side view). For debug purposes.

    Args:
        r (numpy.ndarray): polar radii of photons on detector plane
        phi (numpy.ndarray): polar angles of photons on detector plane
        r0 (numpy.ndarray): initial photons polar radii
        phi0 (numpy.ndarray): initial photons polar angles
        theta0 (numpy.ndarray): initial photons incidence angles
        T (tuple): tuple of ray transfer matrices
        ind (tuple): tuple of photons indices to exclude
        camera (Camera): used camera
    """

    def _generate_color_maps():
        colors1 = LinearSegmentedColormap.from_list("colors", [(0, 0, 0), (0.5, 0.0, 1.0)], N=100)
        colors2 = rainbow(np.linspace(0, 1, 256))
        clr = np.vstack((colors1(np.linspace(0, 1, 100)), colors2))
        return (LinearSegmentedColormap.from_list("colormap", clr),
                LinearSegmentedColormap.from_list("colormap_r", clr[::-1]))

    arr0, xe0, ye0 = _photons2pixels(r0, phi0, camera.image_dims)
    arr, xe, ye = _photons2pixels(r, phi, camera.image_dims)
    plane_cmap, ray_cmap = _generate_color_maps()
    _, ax = plt.subplots(1, 2)
    ax[0].imshow(
        arr0,
        interpolation="nearest",
        origin="low",
        extent=[xe0[0], xe0[-1], ye0[0], ye0[-1]],
        cmap=plane_cmap,
    )
    ax[1].imshow(
        arr,
        interpolation="nearest",
        origin="low",
        extent=[xe[0], xe[-1], ye[0], ye[-1]],
        cmap=plane_cmap,
    )
    _plot_rays(r0, theta0, phi0, T, ind, camera, ray_cmap)
    plt.show(block=False)


def trace(input_plane, camera, show=True):
    """Main method for propagating photons through optical system.

    Propagates light exiting the tissue surface through the optical system to obtain image plane
    data of photons on the optical surface. Optionally can plot the optical system.

    Args:
        input_plane (Path): path to a photon matrix file. Each row of this file is a photon with
            columns: x, y, theta. They designate position and angle with which each photon hits
            optical input plane (tangent to lens surface)
        camera (Camera): used camera
        show (bool): indicates whether or not to generate a visualization of the propagation
            of photons through the system.

    Returns:
        numpy.ndarray: image of the detector plane with photons on it
        numpy.ndarray: x-dimension edges of detector plane
        numpy.ndarray: y-dimension edges of detector plane
    """
    photons = np.genfromtxt(input_plane)
    photons = _convert_photons(photons, camera)
    logger.info('Using input plane data at: %s', input_plane)
    theta0 = photons[:, 2]
    phi0, r0 = _to_polar_coordinates(photons)
    T, r, _, phi, ind = _apply_transform(r0, theta0, phi0, camera)
    r, phi = _correct_angles(r, phi)
    arr, xe, ye = _photons2pixels(r, phi, camera.image_dims)

    if show:
        _plot_photons(r, phi, r0, phi0, theta0, T, ind, camera)

    return arr, xe, ye
