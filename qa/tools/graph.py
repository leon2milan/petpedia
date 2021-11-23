from collections import deque
from digraph import DiGraph


class DiGraph:
    def __init__(self):
        self.graph = {}
        self.vert_count = 0
        self.edge_count = 0

    def add_vertex(self, vertex):
        self.graph[vertex] = set({})
        self.vert_count += 1

    def add_edge(self, start, end):
        if start not in self.graph.keys():
            self.add_vertex(start)
        if end not in self.graph.keys():
            self.add_vertex(end)
        self.graph[start].add(end)
        self.edge_count += 1

    def remove_edge(self, start, end):
        self.graph[start].remove(end)
        self.edge_count -= 1

    def remove_vertex(self, vertex):
        if vertex not in self.graph.keys():
            return
        for _ in self.graph[vertex]:
            self.edge_count -= 1
        for v in self.graph.keys():
            for item in self.graph[v]:
                if item == vertex:
                    self.graph[v].remove(vertex)
                    self.edge_count -= 1
        self.graph.pop(vertex, None)

    def __repr__(self):
        return "DiGraph{" + str(self.graph) + "}"


# PURPOSE
# Given a directed graph in adjacency list format, return a dictionary
# containing the in-degree of each vertex.
# SIGNATURE
# calc_in_degree : DiGraph => Dictionary
# TIME COMPLEXITY
# O(n + m) --each vertex and each edge is visited once.
# SPACE COMPLEXITY
# O(n) --in_degree has one entry per vertex.
def calc_in_degree(g):
    in_degree = {}
    for u in g.graph.keys():
        in_degree[u] = 0
    for u in g.graph.keys():
        for v in g.graph[u]:
            in_degree[v] += 1
    return in_degree


# PURPOSE
# Calculate a topological ordering of a give Digraph in adjacency list format,
# g.
# SIGNATURE
# bfs_topo :: Digraph => List
# TIME COMPLEXITY
# O(n + m) --this is BFS.
# SPACE COMPLEXITY
# O(n) --topological ordering has one entry per vertex, as does in_degree and
# the queue is also O(n)


def bfs_topo(g):
    q = deque()
    topo = []
    in_degree = calc_in_degree(g)
    for u in in_degree.keys():
        if in_degree[u] == 0:
            q.append(u)
    while (q):
        u = q.popleft()
        topo.append(u)
        for v in g.graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
    return topo


# PURPOSE
# Helper function for sssp_dag. Updates the distance to a vertex according
# to whether or not it is cheaper to travel via edge (u, v).
# SIGNATURE
# relax :: Vertex, Vertex, Dictionary, Dictionary => None
# TIME COMPLEXITY
# O(1)
# SPACE COMPLEXITY
# O(1)
def relax(u, v, dist, cost):
    if (u, v) not in cost:
        return
    if dist[u] + cost[(u, v)] < dist[v]:
        dist[v] = dist[u] + cost[(u, v)]


# PURPOSE
# Given a DAG in adjacency list format, g, a starting vertex, s, and a
# dictionary are costs for each edge, calculate the cheapest path from
# s to every other vertex.
# SIGNATURE
# sssp_dag :: Digraph, Vertex, Dictionary => Dictionary
# TIME COMPLEXITY
# O(n + m)--for toposort and the algo itself visits each edge and each vertex.
# SPACE COMPLEXITY
# O(n)--dictionaries, queues, etc. are all O(n).
def sssp_dag(g, s, cost):
    topo_order = bfs_topo(g)
    dist = {}
    for u in g.graph.keys():
        dist[u] = float('inf')
    dist[s] = 0
    # Check each edge to see if it might offer a shorter path.
    for u in topo_order:
        for v in g.graph[u]:
            relax(u, v, dist, cost)
    return dist
