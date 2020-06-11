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

"""Utilities functions."""
import json
import numpy as np


def read_json(json_file):
    """Reads json file content. Returns it as a dict."""
    with open(json_file) as f:
        data = json.load(f)
    return data


def find_nearest(array, value):
    """Finds the nearest match in an array to a user-specified input.

    Args:
        array (array-like): array of values in which to search
        value: value to match to

    Returns:
        type of array element: closest match
        int: index of closest match
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx
