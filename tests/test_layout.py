# -*- coding: utf-8 -*-

"""\
(c) 2015-2018 MGH Computational Pathology

"""
import itertools
from collections import defaultdict

import math

import numpy as np
import nose

from pycrumbs import mock_data, Event, build_tree
from pycrumbs.layout import spring_force, tree_layout, force_adjust


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


def _check_hierarchy(dimensions):
    """Validates that nodes deeper in the tree are lower down in the layout"""
    dim_by_depth = defaultdict(list)
    for dim in dimensions:
        dim_by_depth[dim.node.depth].append(dim)

    # Nodes deeper in the tree should be lower down
    for depth in sorted(dim_by_depth):
        if depth == 0:
            continue

        for dim in dim_by_depth[depth]:
            for higher_up in dim_by_depth[depth - 1]:
                nose.tools.assert_less(dim.node_y, higher_up.node_y)


def test_tree_layout():
    """Tests spatial tree layout"""
    df = mock_data(5000, 500, 4)
    events = Event.from_dataframe(df, 'timestamp', 'observation', 'entity')
    tree = build_tree(events, min_entities_per_node=5)

    dimensions = tree_layout(tree)
    _check_hierarchy(dimensions)

    # No nodes should be less than 1.0 distance unit apart
    for dim_a in dimensions:
        for dim_b in dimensions:
            if dim_a is dim_b:
                continue

            dist = np.linalg.norm(np.asarray([dim_a.node_x, dim_a.node_y]) - np.asarray([dim_b.node_x, dim_b.node_y]))

            nose.tools.assert_greater_equal(dist, 1.0)


def test_spring_force_layout():
    """Tests spring force adjustment"""
    def potential_energy(dim_a, dim_b):
        """Potential energy between two points"""
        return np.linalg.norm(spring_force((dim_a.node_x, dim_b.node_y),
                                           (dim_b.node_x, dim_b.node_y))) ** 2

    def total_energy(dimensions):
        """Computes the total potential energy in the system"""
        energy = 0.0
        dimensions = sorted(dimensions, key=lambda dim: dim.node.depth)
        for _, subdims in itertools.groupby(dimensions, key=lambda dim: dim.node.depth):
            subdims = sorted(subdims, key=lambda dim: dim.node_x)
            for i in range(1, len(subdims)):
                energy += potential_energy(subdims[i - 1], subdims[i])

        return energy

    np.random.seed(0xC0FFEE)
    df = mock_data(5000, 500, 4)
    events = Event.from_dataframe(df, 'timestamp', 'observation', 'entity')
    tree = build_tree(events, min_entities_per_node=5)

    dimensions = tree_layout(tree)
    initial_energy = total_energy(dimensions)

    # The total potential energy in the system should be decreasing
    last_energy = initial_energy
    for n_iter in [50, 100, 200]:
        dimensions_adj = force_adjust(tree_layout(tree), iterations=n_iter, alpha=0.01)
        _check_hierarchy(dimensions_adj)  # It should still look like a tree

        nose.tools.assert_less(total_energy(dimensions_adj), last_energy)
        last_energy = total_energy(dimensions_adj)
