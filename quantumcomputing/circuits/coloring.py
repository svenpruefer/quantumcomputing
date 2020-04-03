# -*- coding: utf-8 -*-

# This file is part of quantumcomputing.
#
# Copyright (c) 2020 by DLR.
from enum import Enum

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit

from circuits.adder import add_full_adder_5, add_full_adder_5_reverse
from circuits.classic import add_xor
from circuits.grover import add_grover_reflection_with_ancilla


class VertexColor(Enum):
    RED = 0
    BLUE = 1
    YELLOW = 2
    GREEN = 3


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
