# -*- coding: utf-8 -*-

# This file is part of quantumcomputing.
#
# Copyright (c) 2020 by DLR.
from enum import Enum
from typing import List, Tuple

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit

from circuits.classic import add_and_4, add_and


class VertexColor(Enum):
    RED = 0  # 00
    BLUE = 1  # 01
    YELLOW = 2  # 10
    GREEN = 3  # 11


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
    if len(ancillas) != 6:
        raise ValueError(f"Need 6 ancillary qubits, but got {len(ancillas)} instead.")
    # Compare the four edges and save their results in the 4 lowest ancillary qubits
    for i, (v1, v2) in enumerate(edges):
        _compare_internal_edge(qc, v1, v2, ancillas[i])
    # Combine the four qubits via an AND operation
    add_and_4(qc, ancillas[0:4], ancillas[4:6], target)
    # Reverse the four edge comparisons to make the ancillary qubits |0> again. Note that the internal edge comparisons
    # all commute, so we don't need to reverse the order.
    for i, (v1, v2) in enumerate(edges):
        _compare_internal_edge(qc, v1, v2, ancillas[i])


def _compare_2_internal_edges(qc: QuantumCircuit, edges: List[Tuple[QuantumRegister, QuantumRegister]],
                              ancillas: List[Qubit], target: Qubit) -> None:
    """
    Compare correctness of two internal edges using two ancillary qubits and save the result into another qubit.

    Note that this circuit is equal to its own reverse circuit.

    :param qc: Underlying QuantumCircuit.
    :param edges: A list of 2-tuples of quantum registers corresponding to the colors of the two vertices bordering that edge.
    :param ancillas: Ancillary qubits to use.
    :param target: If |0> beforehand, this qubit will be |1> if both edges are valid.
    """
    if len(edges) != 2:
        raise ValueError(f"Need 2 edges, but got {len(edges)} instead.")
    if len(ancillas) != 2:
        raise ValueError(f"Need 2 ancillary qubits, but got {len(ancillas)} instead.")
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
    Compare correctness of two external edges using two ancillary qubits and save the result into another qubit.

    Note that this circuit is equal to its own reverse circuit.

    :param qc: Underlying QuantumCircuit.
    :param edges: A list of 2-tuples of quantum registers and vertex colors corresponding to the colors of the two
     vertices bordering that edge.
    :param ancillas: Ancillary qubits to use.
    :param target: If |0> beforehand, this qubit will be |1> if both edges are valid.
    """
    if len(edges) != 2:
        raise ValueError(f"Need 2 edges, but got {len(edges)} instead.")
    if len(ancillas) != 2:
        raise ValueError(f"Need 2 ancillary qubits, but got {len(ancillas)} instead.")
    # Compare the two edges and save their results in the two lowest ancillary qubits
    for i, (v1, v2) in enumerate(edges):
        _compare_external_edge(qc, v1, v2, ancillas[i])
    # Combine the four qubits via an AND operation
    add_and(qc, ancillas[0], ancillas[1], target)
    # Reverse the two edge comparisons to make the ancillary qubits |0> again. Note that the internal edge comparisons
    # all commute, so we don't need to reverse the order.
    for i, (v1, v2) in enumerate(edges):
        _compare_external_edge(qc, v1, v2, ancillas[i])


def _compare_4_external_edges(qc: QuantumCircuit, edges: List[Tuple[QuantumRegister, VertexColor]],
                              ancillas: List[Qubit], target: Qubit) -> None:
    """
    Compare edges of four external vertices using six ancillary qubits and save the result into another qubit.

    Note that this circuit is equal to its own reverse circuit.

    :param qc: Underlying QuantumCircuit.
    :param edges: A list of 2-tuples of quantum registers and vertex color corresponding to the colors of the two
     vertices bordering that edge.
    :param ancillas: Ancillary qubits to use.
    :param target: If |0> beforehand, this qubit will be |1> if all four edges are valid.
    """
    if len(edges) != 4:
        raise ValueError(f"Need 4 edges, but got {len(edges)} instead.")
    if len(ancillas) != 6:
        raise ValueError(f"Need 6 ancillary qubits, but got {len(ancillas)} instead.")
    # Compare the four edges and save their results in the 4 lowest ancillary qubits
    for i, (v1, v2) in enumerate(edges):
        _compare_external_edge(qc, v1, v2, ancillas[i])
    # Combine the four qubits via an AND operation
    add_and_4(qc, ancillas[0:4], ancillas[4:6], target)
    # Reverse the four edge comparisons to make the ancillary qubits |0> again. Note that the internal edge comparisons
    # all commute, so we don't need to reverse the order.
    for i, (v1, v2) in enumerate(edges):
        _compare_external_edge(qc, v1, v2, ancillas[i])
