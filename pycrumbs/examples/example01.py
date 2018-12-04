# -*- coding: utf-8 -*-

"""\
(c) 2015-2018 MGH Computational Pathology

"""

import numpy.random as npr
import matplotlib.pyplot as plt

from pycrumbs import mock_data, Event, build_tree, collapse, pretty_format_tree, draw_tree, new_observation


def main():
    """Main function"""

    # First mock up a dataframe with 40k events from 4000 patients, and with 5 unique types of observations
    df = mock_data(40000, 4000, 5)

    # Convert to pycrumbs Events
    events = Event.from_dataframe(df, 'timestamp', 'observation', 'entity')

    print("{} events".format(len(events)))

    # Build an event transition tree
    tree = build_tree(events, max_depth=10)

    # Collapse nodes with fewer than 25 patients
    tree = collapse(tree, 25)

    # Print the tree
    print(pretty_format_tree(tree))

    # Draw the tree
    draw_tree(tree, cmap='RdYlBu',
              get_color=lambda node: npr.rand(),
              get_edge_label=new_observation,
              iterations=100)
    plt.show()


if __name__ == "__main__":
    main()
