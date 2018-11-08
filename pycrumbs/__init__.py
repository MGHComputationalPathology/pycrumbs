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
    def __init__(self, states, children, entities, name=None, uid=None):
        """

        :param states: Set of states associated with this node, e.g. observations shared by everyone
        in the subtree
        :param children: list of child nodes
        :param entities: list of entities in the subtree
        :param name: name of the node (optional)
        :param uid: unique identifier
        """
        self.state = states
        self.children = children
        self.entities = entities
        self.name = name
        self.uid = uid

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
    def _build(entities, depth):
        unique_states = set(get_state(entity, depth) for entity in entities)
        node = Node(unique_states, [], entities)

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

        node.children = [_build(next_entities, depth + 1)
                         for next_state, next_entities in next_states.items()]
        return node

    entities = Entity.from_events(events)
    root = _build(entities, 0)
    for idx, node in enumerate(root.walk()):
        node.uid = idx

    return root


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


def tree_layout(root, vsep=1.0, hsep=1.0):
    Dimensions = namedtuple("Dimensions", ("x", "y", "width", "height", "node_x", "node_y"))

    def shift_all(dimensions, dx, dy):
        return {uid: Dimensions(dim.x + dx, dim.y + dy, dim.width, dim.height,
                                node_x=dim.node_x + dx, node_y=dim.node_y + dy)
                for uid, dim in dimensions.items()}

    def total_dimensions(dimensions):
        x = min(dim.x for dim in dimensions)
        y = min(dim.y for dim in dimensions)
        max_x = max(dim.x + dim.width for dim in dimensions)
        max_y = max(dim.y + dim.height for dim in dimensions)
        return Dimensions(x, y, max_x - x, max_y - y, None, None)

    def _layout(node):
        if not node.children:
            return {node.uid: Dimensions(0.0, 0.0, 1.0, 1.0, 0.5, 0.5)}

        current_x = 0
        all_dims = {}
        for child in node.children:
            # recursively layout all subtrees
            children_dims = _layout(child)

            # shift horizontally past the previously laid out subtree, and one level down under the root
            children_dims = shift_all(children_dims, current_x, -1.0)

            # calculate bounding box for the subtree
            total_subtree_dim = total_dimensions(children_dims.values())

            current_x += total_subtree_dim.width
            all_dims.update(children_dims)

        total_children_dim = total_dimensions(all_dims.values())

        # Center parent over the children
        all_dims[node.uid] = Dimensions(0.0, 0.0,
                                        total_children_dim.width, total_children_dim.height + 1.0,
                                        total_children_dim.width / 2, 0.0)
        return all_dims

    dims = _layout(root)

    # Center on the nodes and scale to the desired dimensions
    return {uid: (dim.node_x * hsep, dim.node_y * vsep)
            for uid, dim in dims.items()}, dims


def new_observation(parent, child):
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

        return " or ".join(str(", ".join(x)) for x in new_obs)


def draw_tree(tree, get_name=lambda node: len(node.entities),
              get_size=None, get_color=None, cmap=None,
              get_edge_label=None, size=1200.0, draw_bbox=False):

    def default_get_node_size(node):
        return size * math.sqrt(1.0 * len(node.entities) / len(tree.entities))  # area is linearly proportional

    get_size = get_size or default_get_node_size
    node_sizes = {node.uid: get_size(node) for node in tree.walk()}
    node_colors = {node.uid: get_color(node) for node in tree.walk()} if get_color else None

    G = tree.to_graph()
    pos, dimensions = tree_layout(tree)
    nx.draw(G, pos,
            labels={node.uid: get_name(node) for node in tree.walk()},
            node_size=[node_sizes[uid] for uid in G.nodes],
            node_color=([node_colors[uid] for uid in G.nodes] if get_color else None),
            cmap=cmap)

    if draw_bbox:
        ax = plt.gca()
        for _, dim in dimensions.items():
            ax.add_patch(Rectangle((dim.x, dim.y), dim.width, dim.height,
                                  fill=None, alpha=1))

    if get_edge_label:
        nx.draw_networkx_edge_labels(G, pos,
                                     edge_labels={(parent.uid, child.uid): get_edge_label(parent, child)
                                                  for parent, child in tree.walk_edges()})


def main():
    df, meta_df = mock_data(5000, 500, 4)
    events = Event.from_dataframe(df, 'timestamp', 'observation', 'entity')

    print("{} events".format(len(events)))
    tree = build_tree(events, min_entities_per_node=10)
    print(pretty_format_tree(tree))

    draw_tree(tree, cmap='RdYlBu',
              get_color=lambda node: npr.rand(),
              get_edge_label=new_observation)
    plt.show()


if __name__ == "__main__":
    main()
