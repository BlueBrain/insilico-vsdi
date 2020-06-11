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

"""The main pipeline module for generating VSD movies from circuits simulations."""
import logging
import click
from joblib import Parallel, delayed

from insilico_vsdi.utils import read_json
from insilico_vsdi.app import psf_frame, movie_maker, volume
from insilico_vsdi.depth_point_spread import psf

logger = logging.getLogger(__name__)


def _process_single_simulation(sim, config, skip_existing):
    """Produces a VSD movie per simulation.

    Args:
        sim (dict): simulation-related block of parsed VSDI config
        config (dict): parsed VSDI config
        skip_existing (bool): skips processing for output files that already exist
    """
    logger.info('Generating voltage data %s', sim['input'])
    volume.gen_volumes(config, sim['input'], sim['frames-output'])
    logger.info('Generating image frames %s', sim['input'])
    psf_frame.gen_frames(sim['frames-output'], config, skip_existing)
    logger.info('Generating movies %s', sim['input'])
    movie_maker.make_movie(sim['frames-output'], sim['movie-output'], config, skip_existing)


def main(config_path, skip_existing=False):
    """Run the main pipeline.

    It is the sequential run of: volume, psf-frame, movie-maker.

    Args:
        config_path (str): filepath to a VSDI config
        skip_existing (bool): skips processing for output files that already exist
    """
    config = read_json(config_path)
    simulations = config['simulations']
    Parallel()(delayed(_process_single_simulation)(sim, config, skip_existing)
               for sim in simulations)


@click.group('insilico-vsdi')
@click.option('-v', '--verbose', count=True, default=0, help='-v for INFO, -vv for DEBUG')
def cmd_group(verbose):
    """The CLI entry point.

    Args:
        verbose (int): level of logging
    """
    # ERROR level is default to minimize output from neuron_reduce
    level = (logging.WARNING, logging.INFO, logging.DEBUG)[min(verbose, 3)]
    logging.basicConfig(level=level)


@click.command(help=main.__doc__)
@click.option('--config', help='filepath to a VSDI config', required=True)
@click.option('--skip-existing', help='skip processing for output files that already exist',
              is_flag=True)
def cmd(config, skip_existing=False):
    """Cli for `main` function."""
    main(config, skip_existing)


cmd_group.add_command(name='main', cmd=cmd)
cmd_group.add_command(name='volume', cmd=volume.cmd)
cmd_group.add_command(name='psf-frames', cmd=psf_frame.cmd)
cmd_group.add_command(name='movie-maker', cmd=movie_maker.cmd)
cmd_group.add_command(name='generate-psf', cmd=psf.cmd)
