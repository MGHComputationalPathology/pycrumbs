"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""
import math
from collections import defaultdict
from datetime import datetime, timedelta

import networkx as nx

import pandas as pd
import numpy.random as npr

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pycrumbs.layout import tree_layout, force_adjust


def mock_data(n_events, n_entities, n_observations, start_date=None, end_date=None):
    """\
    Creates a synthetic data frame with timestamped observations

    :param n_events: total number of events (#rows in the output)
    :param n_entities: max number of distinct entities
    :param n_observations: max number of distinct types of observations
    :param start_date: start date for the timeseries (default: now)
    :param end_date: end date for the timeseries (default: start_date + 365 days)
    :return: pandas dataframe with columns timestamp, observation, entity

    """
    start_date = start_date or datetime.now()
    end_date = end_date or datetime.now() + timedelta(days=365)
    date_range = end_date - start_date

    if end_date <= start_date:
        raise ValueError("Start date not before end date: {}, {}".format(start_date, end_date))

    obs_space = ['OBS{}'.format(idx) for idx in range(n_observations)]
    entity_space = ['ENT{}'.format(idx) for idx in range(n_entities)]

    event_df = pd.DataFrame()
    event_df['timestamp'] = [start_date + timedelta(seconds=x * date_range.total_seconds()) for x in npr.rand(n_events)]
    event_df['observation'] = npr.choice(obs_space, size=n_events)
    event_df['entity'] = npr.choice(entity_space, size=n_events)

    return event_df


class Event(object):
    """Single timestamped event"""
    def __init__(self, entity, observation, timestamp):
        self.entity = entity
        self.observation = observation
        self.timestamp = timestamp

    @staticmethod
    def from_dataframe(df, time_col, observation_col, entity_col):
        """Parses event objects from a dataframe, returning a list"""
        return Event.from_arrays(df[time_col], df[observation_col], df[entity_col])

    @staticmethod
    def from_arrays(time, observations, entities):
        """Parses event objects from parallel arrays"""
        return [Event(*args) for args in zip(entities, observations, time)]


class Entity(object):
    """An entity with its observations in chronological order"""
    def __init__(self, name, events):
        self.name = name
        self.events = sorted(events, key=lambda obs: obs.timestamp)

    def observations(self, max_events=None):
        """Observations in chronological order, (optionally) up to max_events"""
        if max_events is None:
            max_events = len(max_events)
        elif max_events > len(self.events):
            max_events = len(self.events)

        return tuple(event.observation for event in self.events[:max_events])

    @property
    def frame(self):
        """\
        All events associated with this entity as a data frame, in chronological order. Columns:
        entity, observation, timestamp

        """
        return pd.DataFrame({'entity': obs.entity,
                             'observation': obs.observation,
                             'timestamp': obs.timestamp}
                            for obs in self.events)

    @staticmethod
    def from_events(events):
        """Aggregates events by entity, and returns a list of Entity objects"""
        events_by_entity = defaultdict(list)
        for evt in events:
            events_by_entity[evt.entity].append(evt)
        return [Entity(name, events) for name, events in events_by_entity.items()]


class Node(object):
    """Singl enode in the event tree"""
    def __init__(self, trajectories, children, entities, name=None, uid=None,
                 parent=None):
        """

        :param trajectories: unique trajectories associated with this node
        :param children: list of child nodes
        :param entities: list of entities in the subtree
        :param name: name of the node (optional)
        :param uid: unique identifier
        :param parent: parent node (None if root)
        """
        self.trajectories = trajectories
        self.children = children
        self.entities = entities
        self.name = name
        self.uid = uid
        self.parent = parent

    def walk(self):
        """Depth-first iterator over all nodes in the subtree, including self"""
        yield self
        for child in self.children:
            yield from child.walk()

    def walk_edges(self):
        """Iterates over tuples of all edges in the tree"""
        for child in self.children:
            yield (self, child)
            yield from child.walk_edges()

    def to_graph(self):
        """Converts this subtree into a networkx DiGraph. Uses uid as node names, so make sure to set those."""
        def _recurse(G, node):
            G.add_node(node.uid)
            for child in node.children:
                G.add_edge(node.uid, child.uid)
                _recurse(G, child)
            return G

        G = nx.DiGraph()
        return _recurse(G, self)

    @property
    def depth(self):
        """Depth in the tree (root's depth is 0)"""
        if not self.parent:
            return 0
        return 1 + self.parent.depth


def observations_to_date(entity, depth):
    """Default tree construction criterion: returns the first n event codes, in chronological order"""
    return entity.observations(max_events=depth)


def build_tree(events, get_trajectory=observations_to_date, min_entities_per_node=1, max_depth=None):
    """\
    Builds an event tree

    :param events: list of events
    :param get_trajectory: callable entity, depth => list. Gets the trajectory at the given tree depth. \
    Default: observations_to_date
    :param min_entities_per_node: minimum number of entities per node
    :param max_depth: maximum depth of the tree
    :return: root of the constructed tree

    """
    def _build(entities, depth, parent):
        """Utility: builds a tree from a list of entities"""
        unique_trajectories = set(get_trajectory(entity, depth) for entity in entities)
        node = Node(unique_trajectories, [], entities, parent=parent)

        if min_entities_per_node is not None and len(entities) <= min_entities_per_node:
            return node
        elif max_depth is not None and depth >= max_depth:
            return node

        next_trajectories = defaultdict(list)
        for entity in entities:
            next_trajectories[get_trajectory(entity, depth + 1)].append(entity)

        if len(next_trajectories) == 1 and set(next_trajectories.keys()) == unique_trajectories:
            # No transitions occurred
            return node

        node.children = [_build(next_entities, depth + 1, parent=node)
                         for _, next_entities in next_trajectories.items()]
        return node

    entities = Entity.from_events(events)
    root = _build(entities, 0, None)
    for idx, node in enumerate(root.walk()):
        node.uid = idx

    return root


def collapse(node, min_entities_per_node):
    """\
    Collapses small nodes into a single "other" node at each level. This operation creates a copy
    of the tree.

    :param node: root node/subtree
    :param min_entities_per_node: minimum number of entities per node
    :return: transformed tree (new object)

    """
    if not node.children:
        return node

    new_node = Node(node.trajectories, [],
                    node.entities, node.name, node.uid, node.parent)

    small_children = [child for child in node.children if len(child.entities) < min_entities_per_node]
    large_children = [child for child in node.children if len(child.entities) >= min_entities_per_node]

    if not large_children:  # collapsing would create a single child, so get rid of children altogether
        return new_node

    new_children = [collapse(child, min_entities_per_node) for child in large_children]
    if small_children:
        other_trajectories = set()
        other_entities = []
        for child in small_children:
            other_trajectories.update(child.trajectories)
            other_entities += child.entities

        new_children.append(Node(other_trajectories, [], other_entities, name="Other",
                                 uid=small_children[0].uid, parent=new_node))
    new_node.children = new_children

    return new_node


def pretty_format_tree(root, indent=4):
    """\
    Pretty formats a tree, returning a string

    :param root: root of the subtree to format
    :param indent: indentation between levels (default 4)
    :return: pretty tree as string

    """
    def _format(node, parent, current_indent):
        txt = ""
        txt += "{}* n={} ({:.1%}). Trajectories: {}\n".format(" " * current_indent, len(node.entities),
                                                              1.0 * len(node.entities) / len(parent.entities),
                                                              ", ".join(str(traj) for traj in node.trajectories))

        txt += "".join(_format(child, node, current_indent + indent)
                       for child in node.children)
        return txt

    return _format(root, root, 0).strip()


def new_observation(parent, child, max_observations=1):
    """\
    Gets the new observations between a parent and its children
    :param parent: parent node
    :param child: child node
    :param max_observations: for nodes with multiple associated trajectories, the new "observation" will be a union
    e.g. "OBS1 or OBS2 or OBS3". max_observations specifies the max number of alternatives before the result
    is abbreviated as "other". Default 1

    """
    def begins_with(lst, prefix):
        """Checks whether a list/tuple is a prefix of another"""
        if len(lst) < len(prefix):
            return False
        return lst[:len(prefix)] == prefix

    if len(parent.trajectories) > 1:  # The parent has entities with different trajectories
        return ""
    else:
        parent_trajectory, = parent.trajectories

        new_obs = set()
        for trajectory in child.trajectories:
            if begins_with(trajectory, parent_trajectory):
                new_obs.add(trajectory[len(parent_trajectory):])

        if len(new_obs) > max_observations:
            return "other"
        else:
            return " or ".join(str(", ".join([str(elt) for elt in x])) for x in new_obs)


# pylint: disable=too-many-arguments,too-many-locals
def draw_tree(tree, get_name=lambda node: len(node.entities),
              get_size=None, get_color=None, cmap=None,
              get_edge_label=None, size=1200.0, draw_bbox=False,
              alpha=0.1, iterations=100, spring_constant=0.5, parent_multiplier=10.0,
              neutral_length=1.0):
    """\
    Draws the tree

    :param tree: root/subtree to draw
    :param get_name: label to use for nodes. callable node -> string. Default: number of entities
    :param get_size: node size. callable node -> float. Default: area-proportional to number of entities
    :param get_color: get node color. callable node -> color. Default: None
    :param cmap: color map to use, e.g. RdYlBu
    :param get_edge_label: gets label for edges. callable parent, child -> string. Default: None
    :param size: root node size. Only relevant if using default get_size.
    :param draw_bbox: true to draw bounding boxes around subtrees
    :param alpha: step size of force layout. Larger values will result in larger adjustments.
    :param iterations: number of iterations for force layout. Use 0 to skip force adjustment.
    :param spring_constant: spring constant for force layout.
    :param parent_multiplier: strength multiplier for force from parent
    :param neutral_length: neutral spring length. Default 1.0
    :return: None

    """

    def default_get_node_size(node):
        """Default node size: area-preserving wrt number of entities"""
        return size * math.sqrt(1.0 * len(node.entities) / len(tree.entities))  # area is linearly proportional

    get_size = get_size or default_get_node_size
    node_sizes = {node.uid: get_size(node) for node in tree.walk()}
    node_colors = {node.uid: get_color(node) for node in tree.walk()} if get_color else None

    G = tree.to_graph()
    dimensions = tree_layout(tree)
    if iterations > 0:
        force_adjust(dimensions, alpha=alpha, iterations=iterations, spring_constant=spring_constant,
                     parent_multiplier=parent_multiplier, neutral_length=neutral_length)

    pos = {dim.node.uid: (dim.node_x, dim.node_y)
           for dim in dimensions}

    nx.draw(G, pos,
            labels={node.uid: get_name(node) for node in tree.walk()},
            node_size=[node_sizes[uid] for uid in G.nodes],
            node_color=([node_colors[uid] for uid in G.nodes] if get_color else None),
            cmap=cmap)

    if draw_bbox:
        ax = plt.gca()
        for dim in dimensions:
            ax.add_patch(Rectangle((dim.x, dim.y), dim.width, dim.height,
                                   fill=None, alpha=1))

    if get_edge_label:
        nx.draw_networkx_edge_labels(G, pos,
                                     edge_labels={(parent.uid, child.uid): get_edge_label(parent, child)
                                                  for parent, child in tree.walk_edges()})
