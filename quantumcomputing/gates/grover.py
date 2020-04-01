# -*- coding: utf-8 -*-

# This file is part of quantumcomputing.
#
# Copyright (c) 2020 by DLR.

from qiskit import QuantumCircuit, QuantumRegister
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


def add_grover_with_ancilla_1_0(qc: QuantumCircuit, q_0: Qubit, q_1: Qubit, ancilla: Qubit,
                                measure_ancilla: Clbit) -> None:
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


def add_grover_reflection(qc: QuantumCircuit, register: QuantumRegister) -> None:
    """
    Add a Grover reflection (or amplitude amplification) acting on a specified QuantumRegister in a QuantumCircuit.
    The created circuit will look like the following::

    |            ┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐
    |input_0: |0>┤ H ├┤ X ├┤ H ├┤ X ├┤ H ├┤ X ├┤ H ├
    |            ├───┤├───┤└───┘└─┬─┘└───┘├───┤├───┤
    |input_1: |0>┤ H ├┤ X ├───────■───────┤ X ├┤ H ├
    |            ├───┤├───┤       │       ├───┤├───┤
    |input_2: |0>┤ H ├┤ X ├───────■───────┤ X ├┤ H ├
    |            ├───┤├───┤       │       ├───┤├───┤
    |input_3: |0>┤ H ├┤ X ├───────■───────┤ X ├┤ H ├
    |            ├───┤├───┤       │       ├───┤├───┤
    |input_4: |0>┤ H ├┤ X ├───────■───────┤ X ├┤ H ├
    |            └───┘└───┘               └───┘└───┘

    :param qc: Underlying QuantumCircuit.
    :param register: QuantumRegister containing the qubits whose amplitudes shall be amplified.
    """
    qc.h(register)
    qc.x(register)
    qc.h(register[0])
    qc.mct(q_controls=register[1:], q_target=register[0], q_ancilla=None, mode='noancilla')
    qc.h(register[0])
    qc.x(register)
    qc.h(register)
