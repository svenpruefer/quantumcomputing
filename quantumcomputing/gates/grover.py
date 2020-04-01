# -*- coding: utf-8 -*-

# This file is part of quantumcomputing.
#
# Copyright (c) 2020 by DLR.

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit, Clbit


def _add_2_bit_oracle_without_ancilla_1_0(qc: QuantumCircuit, register: QuantumRegister) -> None:
    """
    Add a Grover oracle for the state |10> on a 2-qubit QuantumRegister.
    When acting on a completely mixed state in this register, this will flip the amplitude of the state |10>.
    The circuit will look like the following::

    | TODO

    :param qc: Underlying QuantumCircuit.
    :param register: QuantumRegister containing exactly two qubits with the qubits ordered in the usual ascending order.
    """
    if len(list(register)) != 2:
        raise ValueError(f"Need QuantumRegister with exactly 2 qubits, but got {len(list(register))} instead.")

    qc.x(register[0])
    qc.cz(register[1], register[0])
    qc.x(register[0])


def _add_grover_step_without_ancilla_1_0(qc: QuantumCircuit, register: QuantumRegister) -> None:
    """
    Add a Grover algorithm step finding the state |10> on a given QuantumRegister with two qubits.
    The circuit will look like the following::

    | TODO

    :param qc: Underlying QuantumCircuit.
    :param register: QuantumRegister containing exactly two qubits with the qubits orderd in the usual ascending order.
    :return:
    """
    if len(list(register)) != 2:
        raise ValueError(f"Need QuantumRegister with exactly 2 qubits, but got {len(list(register))} instead.")

    _add_2_bit_oracle_without_ancilla_1_0(qc, register)  # Oracle
    _add_grover_reflection(qc, register)  # Reflection


def add_grover_without_ancilla_1_0(qc: QuantumCircuit, register: QuantumRegister) -> None:
    """
    Add a Grover algorithm finding the state |10> on a given QuantumRegister with two qubits.
    The circuit will look like the following::

    | TODO

    :param qc: Underlying QuantumCircuit.
    :param register: QuantumRegister containing exactly two qubits with the qubits orderd in the usual ascending order.
    """
    if len(list(register)) != 2:
        raise ValueError(f"Need QuantumRegister with exactly 2 qubits, but got {len(list(register))} instead.")

    qc.h(register)  # Mix states initially
    _add_grover_step_without_ancilla_1_0(qc, register)  # Repeat Grover iteration just once


def add_grover_with_ancilla_1_0(qc: QuantumCircuit, register: QuantumRegister, ancilla: Qubit) -> None:
    """
    Add a Grover algorithm finding the state |10> on a given QuantumRegister with two qubits using a single ancillary qubit.
    The circuit will look like the following::

    | TODO

    :param qc: Underlying QuantumCircuit.
    :param register: QuantumRegister containing exactly two qubits with the qubits orderd in the usual ascending order.
    :param ancilla: Ancillary Qubit.
    """
    if len(list(register)) != 2:
        raise ValueError(f"Need QuantumRegister with exactly 2 qubits, but got {len(list(register))} instead.")

    # Mark states with a 1 at a later determined position by flipping their amplitude
    qc.x(ancilla)
    qc.h(ancilla)
    # Mix input states
    qc.h(register)
    # Execute Grover iteration just once
    _add_grover_step_with_ancilla_1_0(qc, register, ancilla)


def _add_grover_step_with_ancilla_1_0(qc: QuantumCircuit, register: QuantumRegister, ancilla: Qubit) -> None:
    """
    Add a Grover algorithm step finding the state |10> on a given QuantumRegister with two qubits.
    The circuit will look like the following::

    | TODO

    :param qc: Underlying QuantumCircuit.
    :param register: QuantumRegister containing exactly two qubits with the qubits orderd in the usual ascending order.
    :return:
    """
    if len(list(register)) != 2:
        raise ValueError(f"Need QuantumRegister with exactly 2 qubits, but got {len(list(register))} instead.")

    _add_2_bit_oracle_with_ancilla_1_0(qc, register, ancilla)  # Oracle
    _add_grover_reflection(qc, register)  # Reflection


def _add_2_bit_oracle_with_ancilla_1_0(qc: QuantumCircuit, register: QuantumRegister, ancilla: Qubit) -> None:
    """
    Add a Grover oracle for the state |10> on a 2-qubit QuantumRegister using a 1-qubit ancilla.
    When acting on a completely mixed state in this register, this will flip the amplitude of the state |10>.
    The circuit will look like the following::

    | TODO

    :param qc: Underlying QuantumCircuit.
    :param register: QuantumRegister containing exactly two qubits with the qubits ordered in the usual ascending order.
    :param ancilla: Ancillary qubit.
    """
    if len(list(register)) != 2:
        raise ValueError(f"Need QuantumRegister with exactly 2 qubits, but got {len(list(register))} instead.")

    qc.x(register[0])
    qc.ccx(register[1], register[0], ancilla)
    qc.x(register[0])


def _add_grover_reflection(qc: QuantumCircuit, register: QuantumRegister) -> None:
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
