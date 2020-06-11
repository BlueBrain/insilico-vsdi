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

"""Supplementary code for the calculation of a depth-dependent "point-spread function" (PSF).

PSF specifies the amount of spread experienced by point sources of light within the tissue
volume as they propagate up through the microcircuit and the microscope, before landing on the
detector surface.  This code is not necessary to run the pipeline provided a PSF is available, but
may be useful for users who don't already have this information a priori.
"""
