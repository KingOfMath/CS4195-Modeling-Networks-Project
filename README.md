## DataSet

* http://snap.stanford.edu/data/
  * CollegeMsg
* http://konect.uni-koblenz.de
  * Dutch college
  * MovieLens 100k (with rate)

## Sprints

* novel methods
  * random walk algorithm (random find path and traverse)
  * spanning tree (treelib)
  * adjacency matrix
* visualize
  * animate graph generation according to the timestamps
* gain insights
  * recommendation
  * ranking
  * clustering
  
## Metrics
* degree

## Tools

* tacoma
* networkx

## TemporalNetworkWrapper class

Attributes:
* self.**nodes**: list of nodes
* self.**edges**: list of edges formatted as (n1, n2, w, ts)
* self.**temporal_edges**: list of edges list. In **temporal_edges[i]** are stored all edges formatted as **[n1,n2,w] with timestamp i**.
* self.**timestamps**: list of ordered timestamps

**network.json**:
* ["edges"]: contains temporal_edges. All temporal network could be described by tempora_edges list.
* ["timestamps"]: contains custom ordered timestamps list. If it is not specified, standard timestamp starts from 0 to N = len(temporal_edges)

### Init temporal network
``` python
from temporal_network import TemporalNetworkWrapper
tn = TemporalNetworkWrapper() # empty temporal_network
tn = TemporalNetworkWrapper("network.json") # if exists, initialize from json
tn = TemporalNetworkWrapper("network.json", "ral.rating") # if doesn't exist, read from ral.rating and save in network.json
```
