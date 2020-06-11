#!/usr/bin/env python

import imp

from setuptools import setup, find_packages

VERSION = imp.load_source("", "insilico_vsdi/version.py").__version__

with open('README.rst') as f:
    README = f.read()

setup(
    name="insilico-vsdi",
    author="Blue Brain Project, EPFL",
    version=VERSION,
    description="Insilico Voltage Sensitive Dye Imaging",
    long_description=README,
    long_description_content_type='text/x-rst',
    url="https://github.com/BlueBrain/insilico-vsdi",
    license="LGPLv3",
    install_requires=[
        'click>=7.0',
        'matplotlib>=1.3.1',
        'joblib>=0.14',
        'scipy>=1.2',
        'numpy>=1.14',
        'natsort>=5.0',
        'mhd_utils @ git+https://git@github.com/yanlend/mhd_utils.git@master',
    ],
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'insilico_vsdi': ['insilico_vsdi/data/*.*'],
    },
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    entry_points={
        'console_scripts': ['insilico-vsdi=insilico_vsdi.app.__main__:cmd_group'],
    },
)
