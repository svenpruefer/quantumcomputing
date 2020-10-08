# -*- coding: utf-8 -*-

# This file is part of qsoc.
#
# Copyright (c) 2020 by DLR.
from enum import Enum
from typing import List, Tuple, Dict, Set

from more_itertools import grouper
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit

from qsoc.circuits.grover import add_grover_reflection_with_ancilla_on_registers
from qsoc.circuits.classic import add_and_4, add_and, add_and_3


class VertexColor(Enum):
    RED = 0  # 00
    BLUE = 1  # 01
    YELLOW = 2  # 10
    GREEN = 3  # 11


def get_color_from_binary_string(input: str) -> VertexColor:
    if input == "00":
        return VertexColor.RED
    elif input == "01":
        return VertexColor.BLUE
    elif input == "10":
        return VertexColor.YELLOW
    elif input == "11":
        return VertexColor.GREEN
    else:
        raise ValueError(f"Cannot interpret binary string {input} as color")


def _compare_internal_edge(qc: QuantumCircuit, first: QuantumRegister, second: QuantumRegister, target: Qubit) -> None:
    """
    Compare colors of two neighboring vertices along an edge and save the result into another qubit.

    Note that this circuit is equal to its own reverse circuit.

    :param qc: Underlying QuantumCircuit.
    :param first: First quantum register with two qubits encoding a vertex color to compare with `second`.
    :param second: Second quantum register with two qubits encoding a vertex color to compare with `first`.
    :param target: If |0> beforehand, this qubit will be |0> if the colors are the same and |1> if they are different.
    """
    if len(list(first)) != 2:
        raise ValueError(f"First quantum register must have two qubits but has {len(list(first))}")
    if len(list(second)) != 2:
        raise ValueError(f"Second quantum register must have two qubits but has {len(list(second))}")
    qc.cx(first[0], second[0])
    qc.cx(first[1], second[1])
    qc.cx(second[0], target)
    qc.cx(second[1], target)
    qc.ccx(second[0], second[1], target)
    qc.cx(first[1], second[1])
    qc.cx(first[0], second[0])


def _compare_external_edge(qc: QuantumCircuit, vertex: QuantumRegister, external_vertex_color: VertexColor,
                           target: Qubit) -> None:
    """
    Compare colors of an internal vertex with an externally specified color and save the result into another qubit.

    Note that this circuit is equal to its own reverse circuit.

    :param qc: Underlying QuantumCircuit.
    :param vertex: Quantum register with two qubits encoding a vertex color to compare with `external_vertex_color`
    :param external_vertex_color:
    :param target: If |0> beforehand, this qubit will be |0> if the colors are the same and |1> if they are different.
    """
    # Interpret binary representation of colors as qubits
    if external_vertex_color == VertexColor.RED or external_vertex_color == VertexColor.BLUE:
        qc.x(vertex[1])
    if external_vertex_color == VertexColor.RED or external_vertex_color == VertexColor.YELLOW:
        qc.x(vertex[0])
    # AND condition on qubits
    qc.ccx(vertex[1], vertex[0], target)
    # Reverse modifications of qubit inputs
    if external_vertex_color == VertexColor.RED or external_vertex_color == VertexColor.YELLOW:
        qc.x(vertex[0])
    if external_vertex_color == VertexColor.RED or external_vertex_color == VertexColor.BLUE:
        qc.x(vertex[1])
    # Negate result such that 1 corresponds to correct condition
    qc.x(target)


def _compare_4_internal_edges(qc: QuantumCircuit, edges: List[Tuple[QuantumRegister, QuantumRegister]],
                              ancillas: List[Qubit], target: Qubit) -> None:
    """
    Compare colors of four internal vertices using six ancillary qubits and save the result into another qubit.

    Note that this circuit is equal to its own reverse circuit.

    :param qc: Underlying QuantumCircuit.
    :param edges: A list of 2-tuples of quantum registers corresponding to the colors of the two vertices bordering that edge.
    :param ancillas: Ancillary qubits to use.
    :param target: If |0> beforehand, this qubit will be |1> if all four edges are valid.
    """
    if len(edges) != 4:
        raise ValueError(f"Need 4 edges, but got {len(edges)} instead.")
    if len(ancillas) < 6:
        raise ValueError(f"Need at least 6 ancillary qubits, but got {len(ancillas)} instead.")
    # Compare the four edges and save their results in the 4 lowest ancillary qubits
    for i, (v1, v2) in enumerate(edges):
        _compare_internal_edge(qc, v1, v2, ancillas[i])
    # Combine the four qubits via an AND operation
    add_and_4(qc, ancillas[0:4], ancillas[4:6], target)
    # Reverse the four edge comparisons to make the ancillary qubits |0> again. Note that the internal edge comparisons
    # all commute, so we don't need to reverse the order.
    for i, (v1, v2) in enumerate(edges):
        _compare_internal_edge(qc, v1, v2, ancillas[i])


def _compare_3_internal_edges(qc: QuantumCircuit, edges: List[Tuple[QuantumRegister, QuantumRegister]],
                              ancillas: List[Qubit], target: Qubit) -> None:
    """
    Compare colors of three internal vertices using at least four ancillary qubits and save the result into another qubit.

    Note that this circuit is equal to its own reverse circuit.

    :param qc: Underlying QuantumCircuit.
    :param edges: A list of 2-tuples of quantum registers corresponding to the colors of the two vertices bordering that edge.
    :param ancillas: Ancillary qubits to use.
    :param target: If |0> beforehand, this qubit will be |1> if all four edges are valid.
    """
    if len(edges) != 3:
        raise ValueError(f"Need 3 edges, but got {len(edges)} instead.")
    if len(ancillas) < 4:
        raise ValueError(f"Need at least 4 ancillary qubits, but got {len(ancillas)} instead.")
    # Compare the three edges and save their results in the 3 lowest ancillary qubits
    for i, (v1, v2) in enumerate(edges):
        _compare_internal_edge(qc, v1, v2, ancillas[i])
    # Combine the three qubits via an AND operation
    add_and_3(qc, ancillas[0:3], ancillas[3], target)
    # Reverse the three edge comparisons to make the ancillary qubits |0> again. Note that the internal edge comparisons
    # all commute, so we don't need to reverse the order.
    for i, (v1, v2) in enumerate(edges):
        _compare_internal_edge(qc, v1, v2, ancillas[i])


def _compare_2_internal_edges(qc: QuantumCircuit, edges: List[Tuple[QuantumRegister, QuantumRegister]],
                              ancillas: List[Qubit], target: Qubit) -> None:
    """
    Compare correctness of two internal edges using at least two ancillary qubits and save the result into another qubit.

    Note that this circuit is equal to its own reverse circuit.

    :param qc: Underlying QuantumCircuit.
    :param edges: A list of 2-tuples of quantum registers corresponding to the colors of the two vertices bordering that edge.
    :param ancillas: Ancillary qubits to use.
    :param target: If |0> beforehand, this qubit will be |1> if both edges are valid.
    """
    if len(edges) != 2:
        raise ValueError(f"Need 2 edges, but got {len(edges)} instead.")
    if len(ancillas) < 2:
        raise ValueError(f"Need at least 2 ancillary qubits, but got {len(ancillas)} instead.")
    # Compare the two edges and save their results in the two lowest ancillary qubits
    for i, (v1, v2) in enumerate(edges):
        _compare_internal_edge(qc, v1, v2, ancillas[i])
    # Combine the four qubits via an AND operation
    add_and(qc, ancillas[0], ancillas[1], target)
    # Reverse the two edge comparisons to make the ancillary qubits |0> again. Note that the internal edge comparisons
    # all commute, so we don't need to reverse the order.
    for i, (v1, v2) in enumerate(edges):
        _compare_internal_edge(qc, v1, v2, ancillas[i])


def _compare_2_external_edges(qc: QuantumCircuit, edges: List[Tuple[QuantumRegister, VertexColor]],
                              ancillas: List[Qubit], target: Qubit) -> None:
    """
    Compare correctness of two external edges using at least two ancillary qubits and save the result into another qubit.

    Note that this circuit is equal to its own reverse circuit.

    :param qc: Underlying QuantumCircuit.
    :param edges: A list of 2-tuples of quantum registers and vertex colors corresponding to the colors of the two
     vertices bordering that edge.
    :param ancillas: Ancillary qubits to use.
    :param target: If |0> beforehand, this qubit will be |1> if both edges are valid.
    """
    if len(edges) != 2:
        raise ValueError(f"Need 2 edges, but got {len(edges)} instead.")
    if len(ancillas) < 2:
        raise ValueError(f"Need at least 2 ancillary qubits, but got {len(ancillas)} instead.")
    # Compare the two edges and save their results in the two lowest ancillary qubits
    for i, (v1, v2) in enumerate(edges):
        _compare_external_edge(qc, v1, v2, ancillas[i])
    # Combine the four qubits via an AND operation
    add_and(qc, ancillas[0], ancillas[1], target)
    # Reverse the two edge comparisons to make the ancillary qubits |0> again. Note that the internal edge comparisons
    # all commute, so we don't need to reverse the order.
    for i, (v1, v2) in enumerate(edges):
        _compare_external_edge(qc, v1, v2, ancillas[i])


def _compare_3_external_edges(qc: QuantumCircuit, edges: List[Tuple[QuantumRegister, VertexColor]],
                              ancillas: List[Qubit], target: Qubit) -> None:
    """
    Compare correctness of three external edges using at least four ancillary qubits and save the result into another qubit.

    Note that this circuit is equal to its own reverse circuit.

    :param qc: Underlying QuantumCircuit.
    :param edges: A list of 2-tuples of quantum registers and vertex colors corresponding to the colors of the two
     vertices bordering that edge.
    :param ancillas: Ancillary qubits to use.
    :param target: If |0> beforehand, this qubit will be |1> if both edges are valid.
    """
    if len(edges) != 3:
        raise ValueError(f"Need 3 edges, but got {len(edges)} instead.")
    if len(ancillas) < 4:
        raise ValueError(f"Need at least 4 ancillary qubits, but got {len(ancillas)} instead.")
    # Compare the three edges and save their results in the three lowest ancillary qubits
    for i, (v1, v2) in enumerate(edges):
        _compare_external_edge(qc, v1, v2, ancillas[i])
    # Combine the three qubits via an AND operation
    add_and_3(qc, ancillas[0:3], ancillas[3], target)
    # Reverse the two edge comparisons to make the ancillary qubits |0> again. Note that the internal edge comparisons
    # all commute, so we don't need to reverse the order.
    for i, (v1, v2) in enumerate(edges):
        _compare_external_edge(qc, v1, v2, ancillas[i])


def _compare_4_external_edges(qc: QuantumCircuit, edges: List[Tuple[QuantumRegister, VertexColor]],
                              ancillas: List[Qubit], target: Qubit) -> None:
    """
    Compare edges of four external vertices using at least six ancillary qubits and save the result into another qubit.

    Note that this circuit is equal to its own reverse circuit.

    :param qc: Underlying QuantumCircuit.
    :param edges: A list of 2-tuples of quantum registers and vertex color corresponding to the colors of the two
     vertices bordering that edge.
    :param ancillas: Ancillary qubits to use.
    :param target: If |0> beforehand, this qubit will be |1> if all four edges are valid.
    """
    if len(edges) != 4:
        raise ValueError(f"Need 4 edges, but got {len(edges)} instead.")
    if len(ancillas) < 6:
        raise ValueError(f"Need at least 6 ancillary qubits, but got {len(ancillas)} instead.")
    # Compare the four edges and save their results in the 4 lowest ancillary qubits
    for i, (v1, v2) in enumerate(edges):
        _compare_external_edge(qc, v1, v2, ancillas[i])
    # Combine the four qubits via an AND operation
    add_and_4(qc, ancillas[0:4], ancillas[4:6], target)
    # Reverse the four edge comparisons to make the ancillary qubits |0> again. Note that the internal edge comparisons
    # all commute, so we don't need to reverse the order.
    for i, (v1, v2) in enumerate(edges):
        _compare_external_edge(qc, v1, v2, ancillas[i])


def add_4_coloring_oracle(qc, vertices: Dict[str, QuantumRegister], internal_edges: Set[Tuple[str, str]],
                          external_edges: Set[Tuple[str, VertexColor]], auxiliary: QuantumRegister,
                          target: QuantumRegister, ancilla: Qubit) -> None:
    """
    Add a 4-color Grover oracle to a quantum circuit.

    Note that the input is assumed to be feasible. TODO Write down what feasibility entails exactly.

    :param qc: Underlying QuantumCircuit.
    :param vertices: Vertices with corresponding quantum registers modeling their colors.
    :param internal_edges: Internal edges.
    :param external_edges: External edges.
    :param auxiliary: Quantum register for auxiliary qubits used for temporary manipulations. Must be |0> initially.
    :param target: Quantum register to save the correctness of edges temporarilly. Must be |0> initially.
    :param ancilla: Ancillary qubit used to flip the phase of correct solutions.
    """
    # Calculate numbers of grouped edges and group internal edges into groups of four (and one with zero to three)
    number_4_groups_internal_edges = len(internal_edges) // 4
    number_4_groups_external_edges = len(external_edges) // 4

    list_internal_edges: List[Tuple[str, str]] = list(internal_edges)
    list_external_edges: List[Tuple[str, VertexColor]] = list(external_edges)

    groups_internal_edges: List[List[Tuple[str, str]]] = list(
        map(list, grouper(list_internal_edges[:number_4_groups_internal_edges * 4], 4))
    )
    if len(list_internal_edges[number_4_groups_internal_edges * 4:]) > 0:
        groups_internal_edges.append(list_internal_edges[number_4_groups_internal_edges * 4:])
    groups_external_edges: List[List[Tuple[str, VertexColor]]] = list(
        map(list, grouper(list_external_edges[:number_4_groups_external_edges * 4], 4))
    )
    if len(list_external_edges[number_4_groups_external_edges * 4:]) > 0:
        groups_external_edges.append(list_external_edges[number_4_groups_external_edges * 4:])

    if len(groups_internal_edges) + len(groups_external_edges) > len(list(target)):
        raise ValueError(f"Need {len(groups_internal_edges) + len(groups_external_edges)} qubits in 'target' register"
                         f"to save results of edge comparisons, but got only {len(list(target))}")

    # Compare groups of edges
    for i, group in enumerate(groups_internal_edges):
        qubit_group_internal: List[Tuple[QuantumRegister, QuantumRegister]] = [(vertices[v1], vertices[v2]) for v1, v2
                                                                               in group]
        qc.barrier(auxiliary)
        if len(qubit_group_internal) == 4:
            _compare_4_internal_edges(qc, qubit_group_internal, list(auxiliary), target[i])
        if len(qubit_group_internal) == 3:
            _compare_3_internal_edges(qc, qubit_group_internal, list(auxiliary), target[i])
        if len(qubit_group_internal) == 2:
            _compare_2_internal_edges(qc, qubit_group_internal, list(auxiliary), target[i])
        if len(qubit_group_internal) == 1:
            _compare_internal_edge(qc, qubit_group_internal[0][0], qubit_group_internal[0][1], target[i])

    # Group external edges into groups of four and save their states in the remaining part of the target register.
    for i, external_group in enumerate(groups_external_edges):
        qubit_group_external: List[Tuple[QuantumRegister, VertexColor]] = [(vertices[v1], v2) for v1, v2 in
                                                                           external_group]
        qc.barrier(auxiliary)
        if len(qubit_group_external) == 4:
            _compare_4_external_edges(qc, qubit_group_external, list(auxiliary), target[i + len(groups_internal_edges)])
        if len(qubit_group_external) == 3:
            _compare_3_external_edges(qc, qubit_group_external, list(auxiliary), target[i + len(groups_internal_edges)])
        if len(qubit_group_external) == 2:
            _compare_2_external_edges(qc, qubit_group_external, list(auxiliary), target[i + len(groups_internal_edges)])
        if len(qubit_group_external) == 1:
            _compare_external_edge(qc, qubit_group_external[0][0], qubit_group_external[0][1], target[i + len(groups_internal_edges)])

    # Combine target register using Multi-Toffoli gate with target the ancilla qubit to flip the phase of
    # searched states.
    qc.mct(q_controls=target, q_target=ancilla, q_ancilla=auxiliary, mode='basic')

    # Reverse external edges by adding identical gates. As only the target gates are modified
    # via a Multi-Toffoli gate we don't need to reverse the order of the comparisons)
    for i, external_group in enumerate(groups_external_edges):
        qubit_group_external: List[Tuple[QuantumRegister, VertexColor]] = [(vertices[v1], v2) for v1, v2 in
                                                                           external_group]
        qc.barrier(auxiliary)
        if len(qubit_group_external) == 4:
            _compare_4_external_edges(qc, qubit_group_external, list(auxiliary), target[i + len(groups_internal_edges)])
        if len(qubit_group_external) == 3:
            _compare_3_external_edges(qc, qubit_group_external, list(auxiliary), target[i + len(groups_internal_edges)])
        if len(qubit_group_external) == 2:
            _compare_2_external_edges(qc, qubit_group_external, list(auxiliary), target[i + len(groups_internal_edges)])
        if len(qubit_group_external) == 1:
            _compare_external_edge(qc, qubit_group_external[0][0], qubit_group_external[0][1], target[i + len(groups_internal_edges)])

    # Reverse internal edges by adding identical gates. As only the target gates are modified
    # via a Multi-Toffoli gate we don't need to reverse the order of the comparisons)
    for i, group in enumerate(groups_internal_edges):
        qubit_group_internal: List[Tuple[QuantumRegister, QuantumRegister]] = [(vertices[v1], vertices[v2]) for v1, v2
                                                                               in group]
        qc.barrier(auxiliary)
        if len(qubit_group_internal) == 4:
            _compare_4_internal_edges(qc, qubit_group_internal, list(auxiliary), target[i])
        if len(qubit_group_internal) == 3:
            _compare_3_internal_edges(qc, qubit_group_internal, list(auxiliary), target[i])
        if len(qubit_group_internal) == 2:
            _compare_2_internal_edges(qc, qubit_group_internal, list(auxiliary), target[i])
        if len(qubit_group_internal) == 1:
            _compare_internal_edge(qc, qubit_group_internal[0][0], qubit_group_internal[0][1], target[i])


def add_4_coloring_grover(qc, vertices: Dict[str, QuantumRegister], internal_edges: Set[Tuple[str, str]],
                          external_edges: Set[Tuple[str, VertexColor]], auxiliary: QuantumRegister,
                          target: QuantumRegister, ancilla: Qubit, repetitions: int) -> None:
    """
    Add a 4-color Grover algorithm circuit to a quantum circuit.

    Note that the input is assumed to be feasible. TODO Write down what feasibility entails exactly.

    :param qc: Underlying QuantumCircuit.
    :param vertices: Vertices with corresponding quantum registers modeling their colors.
    :param internal_edges: Internal edges.
    :param external_edges: External edges.
    :param auxiliary: Quantum register for auxiliary qubits used for temporary manipulations. Must be |0> initially.
    :param target: Quantum register to save the correctness of edges temporarily. Must be |0> initially.
    :param ancilla: Ancillary qubit used to flip the phase of correct solutions.
    """
    # Mix vertex states
    for vertex, register in vertices.items():
        qc.h(register)

    # Flip phase of ancilla qubit
    qc.x(ancilla)
    qc.h(ancilla)

    # Repeat Grover algorithm
    for i in range(0, repetitions):
        add_4_coloring_oracle(qc, vertices, internal_edges, external_edges, auxiliary, target, ancilla)
        add_grover_reflection_with_ancilla_on_registers(qc, set(vertices.values()), set(list(auxiliary) + list(target)))
