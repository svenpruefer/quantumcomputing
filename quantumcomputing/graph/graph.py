# -*- coding: utf-8 -*-

# This file is part of quantumcomputing.
#
# Copyright (c) 2020 by DLR.

from typing import Set, Dict, Tuple

from qiskit import QuantumCircuit, QuantumRegister

from quantumcomputing.circuits.coloring import VertexColor, add_4_coloring_grover


class Graph:
    _vertices: Set[str] = set()
    _internal_vertices: Set[str] = set()
    _external_vertices: Set[str] = set()
    _edges: Set[Tuple[str, str]] = set()
    _internal_edges: Set[Tuple[str, str]] = set()
    _external_edges: Set[Tuple[str, str]] = set()
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
                self._external_vertices.add(vertex)
            else:
                self._internal_vertices.add(vertex)
        # Separate edges
        self._edges = edges
        for edge in edges:
            if edge[0] in given_colors.keys() and edge[1] in given_colors.keys():
                if given_colors[edge[0]] == given_colors[edge[1]]:
                    raise ValueError(
                        f"Invalid input as the given colors are inconsistent since {edge[0]} and {edge[1]}" +
                        f" have an identical color {given_colors[edge[0]]}")
            elif edge[0] in given_colors.keys():
                self._external_edges.add(edge)
                self._colors[edge[0]] = given_colors[edge[0]]
            elif edge[1] in given_colors.keys():
                self._external_edges.add(edge)
                self._colors[edge[1]] = given_colors[edge[1]]
            else:
                self._internal_edges.add(edge)

        print(f"Created graph with {len(self._internal_vertices)} uncolored vertices,"
              f" {len(self._internal_edges)} internal edges and {len(self._external_edges)} external edges")

    def get_4_color_grover_circuit(self, repetitions: int = 5) -> QuantumCircuit:
        """
        Create a 4-color Grover quantum circuit with 'repetitions' many repetitions for this graph.

        Vertex quantum registers are named as 'v-<name>', where <name> is the name of the vertex and
        the ancilla qubit is called 'ancilla'.

        :return: Quantum Circuit with 4-color Grover circuit
        """
        # Create quantum registers for vertices
        vertices: Dict[str, QuantumRegister] = {name: QuantumRegister(2, f"v-{name}") for name in
                                                self._internal_vertices}

        # Create quantum register for ancilla qubit
        ancilla: QuantumRegister = QuantumRegister(1, "ancilla")

        # Determine colors for external edges
        external_edges: Set[Tuple[str, VertexColor]] = set()
        for v1, v2 in self._external_edges:
            if v1 in self._colors.keys():
                external_edges.add((v2, self._colors[v1]))
            if v2 in self._colors.keys():
                external_edges.add((v1, self._colors[v2]))

        # Determine how many target qubits are needed and create suitable quantum register
        # We separate internal and external edges into groups of four and add one qubit for
        # possibly remaining internal and external edges
        internal_number_4_groups, internal_remainder = divmod(len(self._internal_edges), 4)
        external_number_4_groups, external_remainder = divmod(len(external_edges), 4)
        total_number_target_qubits = internal_number_4_groups + external_number_4_groups
        if internal_remainder > 0:
            total_number_target_qubits += 1
        if external_remainder > 0:
            total_number_target_qubits += 1
        target: QuantumRegister = QuantumRegister(total_number_target_qubits, "target")

        # Determine how many auxiliary qubits are needed and create suitable quantum register
        # The number of auxiliary qubits needed is the maximum number of necessary ancilla qubits
        # for any of the used operations. The needs are as follows:
        # - internal 4 edges: 6
        # - internal 3 edges: 4
        # - internal 2 edges: 2
        # - internal 1 edge: 0
        # - external 4 edges: 6
        # - external 3 edges: 4
        # - external 2 edges: 2
        # - external 1 edge: 0
        # - Multi-Toffoli-gate for combining the target qubits via AND: #target - 2
        # - Grover reflection: 2 * #vertices - 3 - #target
        #   (Because after the Grover oracle step, the target qubits are |0> and can be used as ancilla
        #   qubits for the Grover reflection)
        total_number_auxiliary_qubits = max(6,
                                            total_number_target_qubits - 2,
                                            2 * len(vertices) - 3 - total_number_target_qubits
                                            )
        auxiliary: QuantumRegister = QuantumRegister(total_number_auxiliary_qubits, "auxiliary")

        # Create QuantumCircuit including all quantum registers
        qc: QuantumCircuit = QuantumCircuit(name="four-color-grover-circuit")
        for register in vertices.values():
            qc.add_register(register)
        qc.add_register(auxiliary)
        qc.add_register(target)
        qc.add_register(ancilla)

        print(
            f"Created quantum circuit for graph with {len(vertices)} vertex registers,"
            f" {total_number_auxiliary_qubits} auxiliary qubits,"
            f" {total_number_target_qubits} target qubits and {len(list(ancilla))} ancilla qubits")

        # Add 4-color Grover circuit
        add_4_coloring_grover(qc, vertices, self._internal_edges, external_edges, auxiliary, target, ancilla[0],
                              repetitions)

        return qc
