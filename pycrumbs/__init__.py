"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""
import math
from collections import defaultdict, namedtuple
from datetime import datetime, timedelta

import networkx as nx

import numpy as np
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

    # Generate some random metadata
    age_coef = 2.0 * (npr.rand(len(obs_space)) - 0.5)
    surv_coef = 2.0 * (npr.rand(len(obs_space)) - 0.5)

    meta_records = []
    for entity, sdf in event_df.groupby('entity'):
        counts = sdf['observation'].value_counts()
        obs_val = np.asarray([counts.get(obs, 0) for obs in obs_space])
        meta_records.append({'entity': entity,
                             'age': np.sum(age_coef * obs_val),
                             'survival': np.sum(surv_coef * obs_val)})

    meta_df = pd.DataFrame(meta_records)
    meta_df['age'] = meta_df['age'].min() + meta_df['age']
    meta_df['survival'] = meta_df['survival'].min() + meta_df['survival']
    return event_df, meta_df.set_index('entity')


class Event(object):
    """Single timestamped event"""
    def __init__(self, entity, observation, timestamp):
        self.entity = entity
        self.observation = observation
        self.timestamp = timestamp

    @staticmethod
    def from_dataframe(df, time_col, observation_col, entity_col):
        return Event.from_arrays(df[time_col], df[observation_col], df[entity_col])

    @staticmethod
    def from_arrays(time, observations, entities):
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
    def __init__(self, states, children, entities, name=None, uid=None,
                 parent=None):
        """

        :param states: Set of states associated with this node, e.g. observations shared by everyone
        in the subtree
        :param children: list of child nodes
        :param entities: list of entities in the subtree
        :param name: name of the node (optional)
        :param uid: unique identifier
        :param parent: parent node (None if root)
        """
        self.state = states
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


def build_tree(events, get_state=observations_to_date, min_entities_per_node=1, max_depth=None):
    """\
    Builds an event tree

    :param events:
    :param get_state:
    :param min_entities_per_node:
    :param max_depth:
    :return:
    """
    def _build(entities, depth, parent):
        unique_states = set(get_state(entity, depth) for entity in entities)
        node = Node(unique_states, [], entities, parent=parent)

        if min_entities_per_node is not None and len(entities) <= min_entities_per_node:
            return node
        elif max_depth is not None and depth >= max_depth:
            return node

        next_states = defaultdict(list)
        for entity in entities:
            next_states[get_state(entity, depth + 1)].append(entity)

        if len(next_states) == 1 and set(next_states.keys()) == unique_states:
            # No state transitions occurred
            return node

        node.children = [_build(next_entities, depth + 1, parent=node)
                         for next_state, next_entities in next_states.items()]
        return node

    entities = Entity.from_events(events)
    root = _build(entities, 0, None)
    for idx, node in enumerate(root.walk()):
        node.uid = idx

    return root


def collapse(node, min_entities_per_node):
    if not node.children:
        return node

    new_node = Node(node.state, [],
                    node.entities, node.name, node.uid, node.parent)

    small_children = [child for child in node.children if len(child.entities) < min_entities_per_node]
    large_children = [child for child in node.children if len(child.entities) >= min_entities_per_node]

    if not large_children:  # collapsing would create a single child, so get rid of children altogether
        return new_node

    new_children = [collapse(child, min_entities_per_node) for child in large_children]
    if small_children:
        other_states = set()
        other_entities = []
        for child in small_children:
            other_states.update(child.state)
            other_entities += child.entities

        new_children.append(Node(other_states, [], other_entities, name="Other",
                                 uid=small_children[0].uid, parent=new_node))
    new_node.children = new_children

    return new_node


def pretty_format_tree(root, indent=4):
    def _format(node, parent, current_indent):
        txt = ""
        txt += "{}* n={} ({:.1%}). States: {}\n".format(" " * current_indent, len(node.entities),
                                                        1.0 * len(node.entities) / len(parent.entities),
                                                        ", ".join(str(state) for state in node.state))

        txt += "".join(_format(child, node, current_indent + indent)
                       for child in node.children)
        return txt

    return _format(root, root, 0).strip()


def new_observation(parent, child, max_observations=1):
    def begins_with(lst, prefix):
        if len(lst) < len(prefix):
            return False
        return lst[:len(prefix)] == prefix

    if len(parent.state) > 1:
        return ""
    else:
        parent_state, = parent.state

        new_obs = set()
        for state in child.state:
            if begins_with(state, parent_state):
                new_obs.add(state[len(parent_state):])

        if len(new_obs) > max_observations:
            return "other"
        else:
            return " or ".join(str(", ".join([str(elt) for elt in x])) for x in new_obs)


def draw_tree(tree, get_name=lambda node: len(node.entities),
              get_size=None, get_color=None, cmap=None,
              get_edge_label=None, size=1200.0, draw_bbox=False,
              alpha=0.1, iterations=100, spring_constant=0.5, parent_multiplier=10.0):

    def default_get_node_size(node):
        return size * math.sqrt(1.0 * len(node.entities) / len(tree.entities))  # area is linearly proportional

    get_size = get_size or default_get_node_size
    node_sizes = {node.uid: get_size(node) for node in tree.walk()}
    node_colors = {node.uid: get_color(node) for node in tree.walk()} if get_color else None

    G = tree.to_graph()
    dimensions = tree_layout(tree)
    if iterations > 0:
        force_adjust(dimensions, alpha=alpha, iterations=iterations, spring_constant=spring_constant,
                     parent_multiplier=parent_multiplier)

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


def main():
    np.random.seed(0xC0FFEE)
    df, meta_df = mock_data(5000, 500, 4)
    events = Event.from_dataframe(df, 'timestamp', 'observation', 'entity')

    print("{} events".format(len(events)))
    tree = build_tree(events, min_entities_per_node=20)
    tree = collapse(tree, 20)
    print("Unique depths: {}".format(sorted(set(node.depth for node in tree.walk()))))
    print(pretty_format_tree(tree))

    draw_tree(tree, cmap='RdYlBu',
              get_color=lambda node: npr.rand(),
              get_edge_label=new_observation)
    plt.show()


if __name__ == "__main__":
    main()
