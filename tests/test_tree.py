# -*- coding: utf-8 -*-

"""\
(c) 2015-2018 MGH Computational Pathology

"""
import nose
import numpy as np

from pycrumbs import mock_data, Event, build_tree


def test_tree_construction():
    """Tests construction of transition trees from events"""
    df = mock_data(5000, 500, 4)
    events = Event.from_dataframe(df, 'timestamp', 'observation', 'entity')
    tree = build_tree(events, min_entities_per_node=5)

    # Sanity check: the tree should have a few nodes
    nose.tools.assert_greater(len(list(tree.walk())), 10)

    # All entities should be present at the leaves
    entities = []
    for node in tree.walk():
        if not node.children:
            entities += node.entities

    nose.tools.eq_(len(entities), len(df['entity'].unique()))
    nose.tools.eq_({ent.name for ent in entities}, set(df['entity'].unique()))

    # The events are uniform so the tree should be about 4-5 levels deep (i.e. log4(500))
    mean_leaf_depth = np.mean([node.depth for node in tree.walk() if not node.children])
    nose.tools.assert_greater(mean_leaf_depth, 3)
    nose.tools.assert_less(mean_leaf_depth, 5)

    # Children should have more events than the parent
    for node in tree.walk():
        if not node.children:  # leaf
            continue

        nose.tools.assert_greater(np.mean([len(traj) for child in node.children for traj in child.trajectories]),
                                  np.mean([len(traj) for traj in node.trajectories]))
