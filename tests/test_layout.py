# -*- coding: utf-8 -*-

"""\
(c) 2015-2018 MGH Computational Pathology

"""

from __future__ import unicode_literals
from __future__ import print_function

import math

import numpy as np
import nose

from pycrumbs.layout import spring_force


def test_force_at_neutral():
    """Spring force should be 0 at neutral length"""
    def checkme(pos_a, pos_b, neutral_length, expected):
        np.testing.assert_allclose(spring_force(pos_a, pos_b, neutral_length=neutral_length), expected)

    for neutral_length in [0.1, 1, 10]:
        # At neutral length the forces should be 0
        yield checkme, (0, 0), (neutral_length, 0), neutral_length, (0, 0)
        yield checkme, (0, 0), (0, neutral_length), neutral_length, (0, 0)
        yield checkme, (-neutral_length / 2, 0), (neutral_length / 2, 0), neutral_length, (0, 0)


def test_extension():
    """Extension beyond neutral length: force should be increasing"""
    for step in np.linspace(1.0, 10.0, 100):
        nose.tools.assert_greater(np.linalg.norm(spring_force((0.0, 0.0), (step + 0.01, step + 0.01))),
                                  np.linalg.norm(spring_force((0.0, 0.0), (step, step))))

        nose.tools.assert_greater(np.linalg.norm(spring_force((0.0, 0.0), (step, step + 0.01))),
                                  np.linalg.norm(spring_force((0.0, 0.0), (step, step))))

        nose.tools.assert_greater(np.linalg.norm(spring_force((0.0, 0.0), (step + 0.01, step))),
                                  np.linalg.norm(spring_force((0.0, 0.0), (step, step))))


def test_compression():
    """Compression: force should be increasing with smaller length, i.e. opposite of extension"""
    for step in np.linspace(math.sqrt(2) / 2, 0.02, 100):
        print(step)
        nose.tools.assert_greater(np.linalg.norm(spring_force((0.0, 0.0), (step - 0.01, step - 0.01))),
                                  np.linalg.norm(spring_force((0.0, 0.0), (step, step))))

        nose.tools.assert_greater(np.linalg.norm(spring_force((0.0, 0.0), (step, step - 0.01))),
                                  np.linalg.norm(spring_force((0.0, 0.0), (step, step))))

        nose.tools.assert_greater(np.linalg.norm(spring_force((0.0, 0.0), (step - 0.01, step))),
                                  np.linalg.norm(spring_force((0.0, 0.0), (step, step))))


def test_direction():
    """Validates that the force has the correct direction"""
    def sign(x):
        if x > 0:
            return 1.0
        elif x < 0:
            return -1.0
        else:
            return 0.0

    def checkme(pos_a, pos_b, expected_sign):
        force_vec = spring_force((pos_a, 0), (pos_b, 0))
        if sign(force_vec[0]) != expected_sign:
            raise AssertionError("Incorrect force sign between {} and {}: f={}".format(pos_a, pos_b, force_vec))

    yield checkme, 0.0, 2.0, -1.0  # extended, B is pulled left
    yield checkme, 0.0, 3.0, -1.0  # likewise

    yield checkme, 0.0, 0.5, 1.0  # compressed, B is pushed right
    yield checkme, 0.0, 0.25, 1.0  # likewise
