# -*- coding: utf-8 -*-

# This file is part of quantumcomputing.
#
# Copyright (c) 2020 by DLR.

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit

from circuits.adder import add_full_adder_5, add_full_adder_5_reverse
from circuits.classic import add_xor
from circuits.grover import add_grover_reflection_no_ancilla


def add_max_cut_circuit(qc: QuantumCircuit, vertices: QuantumRegister, edges: QuantumRegister,
                        summation: QuantumRegister, ancilla: Qubit) -> None:
    """
    Add a MAX-CUT circuit to a QuantumCircuit.

    It uses the Grover algorithm to amplify amplitudes of vertices having a number of edges with different
    vertex-colorings specified by the ancilla qubit.

    Note that at the moment it can only find states having exactly three edges with different colored vertices at
    both ends.

    :param qc: Underlying QuantumCircuit.
    :param vertices: Input qubits corresponding to colors of vertices of graph.
    :param edges: Qubits corresponding to edges in graph.
    :param summation: Qubits used for counting of suitable edges.
    :param ancilla: Ancillary qubit to flip phase of target states.
    """
    # Mix initial states
    qc.h(vertices)
    # Prepare ancilla qubit
    qc.x(ancilla)
    qc.h(ancilla)
    # Execute Grover iteration twice
    _add_max_cut_step(qc, vertices, edges, summation, ancilla)
    _add_max_cut_step(qc, vertices, edges, summation, ancilla)


def _add_max_cut_step(qc: QuantumCircuit, vertices: QuantumRegister, edges: QuantumRegister, summation: QuantumRegister,
                      ancilla: Qubit) -> None:
    """
    Add a MAX-CUT Grover step to a QuantumCircuit.

    Note that at the moment it can only find states having exactly three edges with different colored vertices at
    both ends. Also, it works only on the default example graph.

    :param qc: Underlying QuantumCircuit.
    :param vertices: Input qubits corresponding to colors of vertices of graph.
    :param edges: Qubits corresponding to edges in graph.
    :param summation: Qubits used for counting of suitable edges.
    :param ancilla: Ancillary qubit to flip phase of target states.
    """
    _add_max_cut_oracle(qc, vertices, edges, summation, ancilla)
    add_grover_reflection_no_ancilla(qc, vertices)


def _add_max_cut_oracle(qc: QuantumCircuit, vertices: QuantumRegister, edges: QuantumRegister,
                        summation: QuantumRegister, ancilla: Qubit) -> None:
    """
    Add a MAX-CUT Grover step to a QuantumCircuit.

    Note that at the moment it can only find states having exactly three edges with different colored vertices at
    both ends. Also, it works only on the default example graph.

    :param qc: Underlying QuantumCircuit.
    :param vertices: Input qubits corresponding to colors of vertices of graph.
    :param edges: Qubits corresponding to edges in graph.
    :param summation: Qubits used for counting of suitable edges.
    :param ancilla: Ancillary qubit to flip phase of target states.
    """
    # Compare vertex colorings according to graph
    add_xor(qc, vertices[3], vertices[0], edges[2])
    add_xor(qc, vertices[2], vertices[0], edges[1])
    add_xor(qc, vertices[1], vertices[0], edges[0])
    # Add edge counts
    add_full_adder_5(qc, edges[0], edges[1], edges[2], summation[0], summation[1])
    # Entangle ancilla qubit with sums in order to flip phase of target states
    qc.ccx(summation[0], summation[1], ancilla)
    # Reverse edge counts to reset summation qubits
    add_full_adder_5_reverse(qc, edges[0], edges[1], edges[2], summation[0], summation[1])
    # Reverse vertex comparisons to reset edge qubits
    # Note that XOR is symmetric and commutative, so we don't need to reverse them
    add_xor(qc, vertices[3], vertices[0], edges[2])
    add_xor(qc, vertices[2], vertices[0], edges[1])
    add_xor(qc, vertices[1], vertices[0], edges[0])