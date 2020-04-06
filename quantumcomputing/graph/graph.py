# -*- coding: utf-8 -*-

# This file is part of quantumcomputing.
#
# Copyright (c) 2020 by DLR.

from typing import Set, Dict, Tuple

from quantumcomputing.circuits.coloring import VertexColor


class Graph:
    _vertices: Set[str] = set()
    _internal_vertices: Set[str] = set()
    _external_vertices: Set[str] = set()
    _edges: Set[Tuple[str, str]] = set()
    _internal_edges: Set[Tuple[str, str]] = set()
    _external_edges: Set[Tuple[str, VertexColor]] = set()
    _colors: Dict[str, VertexColor] = {}

    def __init__(self, vertices: Set[str], edges: Set[Tuple[str, str]],
                 given_colors: Dict[str, VertexColor]):
        """
        Constructor for graphs for which we want to solve the four-color problem using a quantum computer.
        :param vertices: Set of vertices of the graph. Needs to be less than 12.
        :param edges: Set of edges of the graph. Needs to be less than 25.
        :param given_colors: Any given colors.
        """
        # We consider only simple graphs that are not too large and whose edges and given colors are meaningful.
        if len(vertices) > 11:
            raise ValueError(f"Can only consider graphs with less than 12 vertices, but got {len(vertices)}")
        if len(edges) > 24:
            raise ValueError(f"Can only consider graphs with less than 25 edges, but got {len(edges)}")
        for edge_endpoint in list(sum(edges, ())):
            if edge_endpoint not in vertices:
                raise ValueError(f"Endpoint {edge_endpoint} of an edge is not contained in the set of vertices.")
        for vertex in given_colors.keys():
            if vertex not in vertices:
                raise ValueError(f"Vertex {vertex} with a specified color is not contained in the set of vertices.")
        # Separate vertices
        self._vertices = vertices
        for vertex in vertices:
            if vertex in given_colors.keys():
                self._external_vertices += vertex
            else:
                self._internal_vertices += vertex
        # Separate edges
        self._edges = edges
        for edge in edges:
            if edge[0] in given_colors.keys() and edge[1] in given_colors.keys():
                if given_colors[edge[0]] == given_colors[edge[1]]:
                    raise ValueError(
                        f"Invalid input as the given colors are inconsistent since {edge[0]} and {edge[1]}" +
                        f" have an identical color {given_colors[edge[0]]}")
            elif edge[0] in given_colors.keys():
                self._external_edges += edge
                self._colors[edge[0]] = given_colors[edge[0]]
            elif edge[1] in given_colors.keys():
                self._external_edges += edge
                self._colors[edge[1]] = given_colors[edge[1]]
            else:
                self._internal_edges += edge
