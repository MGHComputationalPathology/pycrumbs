# -*- coding: utf-8 -*-

"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""

from __future__ import unicode_literals
from __future__ import print_function

import re
from distutils.core import setup
from setuptools import find_packages


def read_requirements():
    """Reads the list of required packages from requirements.txt"""
    def parse_one(line):
        """Parses a single requirement"""
        line = line.strip()
        m = re.match(r'-e.*/(.*)@.*#egg=.*', line)  # for git links: use the package name
        if m is not None:
            return m.group(1)
        else:
            return line

    with open('requirements.txt', 'r') as f:
        return [parse_one(line) for line in f if '://' not in line]

with open('version.txt', 'r') as f:
    version = f.read().strip()

setup(
    name='pycrumbs',
    version=version,
    packages=find_packages(),
    install_requires=read_requirements(),
    url='https://github.com/MGHComputationalPathology/pycrumbs',
    license='BSD3',
    author='MGH Computational Pathology',
    author_email='mpacula@mgh.harvard.edu',
    description='Tools for time-series visualization'
)
