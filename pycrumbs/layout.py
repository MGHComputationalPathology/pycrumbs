# -*- coding: utf-8 -*-

"""\
(c) 2015-2018 MGH Computational Pathology


 Force-directed layout for trees

"""
from collections import defaultdict

import numpy as np


class Dimensions:
    def __init__(self, node, x, y, width, height, node_x, node_y):
        self.node = node
        self.x, self.y = x, y
        self.width, self.height = width, height
        self.node_x, self.node_y = node_x, node_y


def spring_force(pos_a, pos_b, neutral_length=1.0, spring_constant=1.0, alpha=0.01):
    """\
    Given the x,y coordinates of two objects, A and B, returns the spring force acting on B as a vector

    """
    if pos_a is None or pos_b is None:
        return np.array([0.0, 0.0])

    length = np.linalg.norm(np.asarray(pos_b) - np.asarray(pos_a))
    unit_vec = (np.asarray(pos_b) - np.asarray(pos_a)) / length

    if length > neutral_length:
        return -unit_vec * spring_constant * (length - neutral_length)
    elif length > 0:
        intercept = 1.0 / (neutral_length + alpha)
        return unit_vec * spring_constant * (1.0 / (length + alpha) - intercept)
    else:
        return (np.random.random() - 0.5) * np.array([spring_constant, spring_constant])  # perturb randomly


def force_adjust(dimensions, alpha=0.1, iterations=100, spring_constant=0.5, parent_multiplier=10.0):
    def adjust_at_depth(dims, min_x, max_x):
        left_anchor = Dimensions(None, None, None, None, None, min_x, 0)
        right_anchor = Dimensions(None, None, None, None, None, max_x, 0)

        dims_by_uid = {dim.node.uid: dim for dim in dims}

        dims = sorted(dims, key=lambda dim: dim.node_x)
        for i in np.random.permutation(len(dims)):
            left = dims[i - 1] if i > 0 else left_anchor
            right = dims[i + 1] if i < len(dims) - 1 else right_anchor

            # Force due to left neighbor
            force_vec = spring_force((left.node_x, left.node_y) if left else None,
                                     (dims[i].node_x, dims[i].node_y),
                                     spring_constant=spring_constant)

            # Force due to right neighbor
            force_vec += spring_force((right.node_x, right.node_y) if right else None,
                                      (dims[i].node_x, dims[i].node_y),
                                      spring_constant=spring_constant)

            # Force due to parent
            parent = dims_by_uid.get(dims[i].node.parent.uid) if dims[i].node.parent else None
            force_vec += parent_multiplier * spring_force((parent.node_x, parent.node_y) if parent else None,
                                                          (dims[i].node_x, dims[i].node_y),
                                                          spring_constant=spring_constant)

            # We put the nodes on rails in the x dimension, i.e. ignore the y component of the force
            dx = force_vec[0] * alpha
            dims[i].x += dx
            dims[i].node_x += dx

    # Collect nodes by depth
    dim_by_depth = defaultdict(list)
    for dim in dimensions:
        dim_by_depth[dim.node.depth].append(dim)

    root_dim, = dim_by_depth[0]
    min_x = root_dim.x
    max_x = root_dim.x + root_dim.width

    for _ in range(iterations):
        for _, dims in dim_by_depth.items():
            adjust_at_depth(dims, min_x, max_x)

    return dimensions


def tree_layout(root):
    def shift_all(dimensions, dx, dy):
        return [Dimensions(dim.node, dim.x + dx, dim.y + dy, dim.width, dim.height,
                           node_x=dim.node_x + dx, node_y=dim.node_y + dy)
                for dim in dimensions]

    def total_dimensions(dimensions):
        x = min(dim.x for dim in dimensions)
        y = min(dim.y for dim in dimensions)
        max_x = max(dim.x + dim.width for dim in dimensions)
        max_y = max(dim.y + dim.height for dim in dimensions)
        return Dimensions(None, x, y, max_x - x, max_y - y, None, None)

    def _layout(node):
        if not node.children:
            return [Dimensions(node, 0.0, 0.0, 1.0, 1.0, 0.5, 0.0)]

        current_x = 0
        all_dims = []
        for child in node.children:
            # recursively layout all subtrees
            children_dims = _layout(child)

            # shift horizontally past the previously laid out subtree, and one level down under the root
            children_dims = shift_all(children_dims, current_x, -1.0)

            # calculate bounding box for the subtree
            total_subtree_dim = total_dimensions(children_dims)

            current_x += total_subtree_dim.width
            all_dims.extend(children_dims)

        total_children_dim = total_dimensions(all_dims)

        # Center parent over the children
        all_dims.append(Dimensions(node, 0.0, 0.0,
                                   total_children_dim.width, total_children_dim.height + 1.0,
                                   total_children_dim.width / 2, 0.0))
        return all_dims

    dims = _layout(root)

    # Center on the nodes and scale to the desired dimensions
    return dims
