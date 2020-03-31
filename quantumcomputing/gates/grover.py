# -*- coding: utf-8 -*-

# This file is part of quantumcomputing.
#
# Copyright (c) 2020 by DLR.

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit, Clbit


def add_grover_without_ancilla_1_0(qc: QuantumCircuit, q_0: Qubit, q_1: Qubit) -> None:
    # Oracle
    qc.x(q_0)
    qc.cz(q_1, q_0)
    qc.x(q_0)
    qc.barrier()
    # Mix
    qc.h(q_0)
    qc.h(q_1)
    qc.barrier()
    # Reflection
    qc.x(q_1)
    qc.x(q_0)
    qc.h(q_0)
    qc.cx(q_1, q_0)
    qc.h(q_0)
    qc.x(q_1)
    qc.x(q_0)
    qc.barrier()
    # Mix
    qc.h(q_0)
    qc.h(q_1)
    qc.barrier()


def add_grover_with_ancilla_1_0(qc: QuantumCircuit, q_0: Qubit, q_1: Qubit, ancilla: Qubit, measure_ancilla: Clbit) -> None:
    # Setup
    qc.x(ancilla)
    qc.h(q_0)
    qc.h(q_1)
    qc.h(ancilla)
    qc.barrier()
    # Oracle
    qc.x(q_0)
    qc.ccx(q_1, q_0, ancilla)
    qc.x(q_0)
    qc.measure(ancilla, measure_ancilla)
    qc.barrier()
    # Mix
    qc.h(q_0)
    qc.h(q_1)
    qc.barrier()
    # Reflection
    qc.x(q_1)
    qc.x(q_0)
    qc.h(q_0)
    qc.cx(q_1, q_0)
    qc.h(q_0)
    qc.x(q_1)
    qc.x(q_0)
    qc.barrier()
    # Mix
    qc.h(q_0)
    qc.h(q_1)
    qc.barrier()
