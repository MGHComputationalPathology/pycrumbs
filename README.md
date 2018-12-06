
# PyCrumbs
[![Build Status](https://travis-ci.org/MGHComputationalPathology/pycrumbs.svg?branch=master)](https://travis-ci.org/MGHComputationalPathology/pycrumbs)


## Introduction
PyCrumbs is a Python library for visualizing trajectories from longitudinal event data. It was designed with healthcare data in mind (e.g. medications & treatment trajectories) but generalizes to other domains as well.

At the base of PyCrumbs is an `Event` representing a single timestamped observation for an "entity" (e.g. patient). The library groups and tracks events across unique entities.

Entities can be transformed into a trajectory tree which represents transitions between observations. At the root, the patients don't have any observations (empty trajectory). They subsequently transition into child nodes based on the type of observation they acquire.

## Installation
```bash
pip install git+https://github.com/MGHComputationalPathology/pycrumbs.git
```

## Example

Let's begin by mocking up 10k observations for 1000 patients, corresponding to 5 discrete types of medications.


```python
from pycrumbs import *

df = mock_data(n_events=10000, n_entities=1000, n_observations=5)
df.head(10)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>timestamp</th>
      <th>observation</th>
      <th>entity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-05-31 21:05:25.355117</td>
      <td>MEDICATION_1</td>
      <td>PATIENT_774</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-02-13 12:05:55.233914</td>
      <td>MEDICATION_0</td>
      <td>PATIENT_344</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-01-27 02:49:00.921409</td>
      <td>MEDICATION_2</td>
      <td>PATIENT_55</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-07-22 10:47:34.983793</td>
      <td>MEDICATION_0</td>
      <td>PATIENT_284</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-09-30 16:27:38.617548</td>
      <td>MEDICATION_3</td>
      <td>PATIENT_130</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2019-05-19 01:37:47.495633</td>
      <td>MEDICATION_3</td>
      <td>PATIENT_459</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2019-06-23 07:45:25.729100</td>
      <td>MEDICATION_1</td>
      <td>PATIENT_899</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2019-02-25 09:34:21.037985</td>
      <td>MEDICATION_2</td>
      <td>PATIENT_838</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2019-03-26 17:02:33.282783</td>
      <td>MEDICATION_4</td>
      <td>PATIENT_214</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2019-03-05 18:35:58.703941</td>
      <td>MEDICATION_2</td>
      <td>PATIENT_909</td>
    </tr>
  </tbody>
</table>
</div>



Now, let's convert the data frame to a list of `Event`s:


```python
events = Event.from_dataframe(df, "timestamp", "observation", "entity")
```

With events in hand, we can build the trajectory tree. Note that root has depth 0, so `max_depth=2` will build a tree with 3 levels.


```python
tree = build_tree(events, max_depth=2, min_entities_per_node=10)
```

The tree can be plotted by calling `draw_tree`. Here, we color nodes randomly and display acquired observations on each edge.


```python
plt.figure(figsize=(15, 10))
draw_tree(tree, 
          get_color=lambda node: npr.rand(),
          get_edge_label=new_observation)
```


![png](examples/output_8_0.png)


The defaults can be easily customized by passing different functions. For example, below we add transition probabilities to each edge.


```python
def my_edge_label(parent, child):
    return "{} (p={:.1%})".format(new_observation(parent, child),
                                  1.0 * len(child.entities) / len(parent.entities))

plt.figure(figsize=(15, 10))
draw_tree(tree, 
          get_color=lambda node: npr.rand(),
          get_edge_label=my_edge_label)
```


![png](examples/output_10_0.png)
