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

"""Module that generates raw volumetric voltage data using EMSim from a circuit simulation."""
from pathlib import Path
import subprocess
import logging
import pkg_resources
import click

from insilico_vsdi.utils import read_json

logger = logging.getLogger(__name__)


def gen_volumes(config, input_, frames_output):
    """Generate raw VSDI data volumes using EMSim.

    The resulting data volumes are already scaled by an attenuation factor accounting for light and
    dye penetration in cortical tissue.

    Args:
        config (dict): parsed VSDI config
        input_ (str): filepath to BlueConfig of the input simulation
        frames_output (str): frames output filepath. The parent directory of these files is used to
            store emsim-vsd results.

    Returns:
        int: returncode of EMSim process.
    """
    params = config['emsim-vsd-args']
    report_area = Path(input_).parent / (params['report-area'] + '.bbp')
    curve = params['curve']
    if curve is None:
        curve = pkg_resources.resource_filename(
            __name__.split('.')[0], 'data/RH1691-cortical-penetration-mouse.txt')

    cmd_list = [
        'emsimVSD',
        '-i', input_,
        '-o', frames_output,
        '--report-voltage', params['report-voltage'],
        '--report-area', report_area,
        '--target', params['target'],
        '--sensor-dim', str(params['sensor-dim']),
        '--sensor-res', str(params['sensor-res']),
        '--start-time', str(params['start-time']),
        '--end-time', str(params['end-time']),
        '--time-step', str(params['time-step']),
        '--sigma', str(params['sigma']),
        '--g0', str(params['g0']),
        '--curve', curve,
        '--ap-threshold', str(params['ap-threshold']),
        '--v0', str(params['v0']),
        '--depth', str(params['depth']),
        '--fraction', str(params['fraction']),
    ]

    if params['export-volume']:
        cmd_list.append('--export-volume')
    if params['interpolate-attenuation']:
        cmd_list.append('--interpolate-attenuation')
    if params['soma-pixels']:
        cmd_list.append('--soma-pixels')

    process = subprocess.run(cmd_list, check=True, timeout=60 * 60)
    return process.returncode


@click.command(help=gen_volumes.__doc__)
@click.option('--config', help='filepath to a VSDI config', required=True)
@click.option('--input', 'input_', help='filepath to BlueConfig of input simulation', required=True)
@click.option('--frames-output', help='frames output filepath', required=True)
def cmd(config, input_, frames_output):
    """Cli for `gen_volumes` function."""
    gen_volumes(read_json(config), input_, frames_output)
