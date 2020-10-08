# -*- coding: utf-8 -*-

# This file is part of qsoc.
#
# Copyright (c) 2020 by DLR.

from typing import List

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit


def add_and(qc: QuantumCircuit, first: Qubit, second: Qubit, target: Qubit) -> None:
    """
    Add an AND gate combining `first` and `second` registers and writing the result into `target`.
    """
    qc.ccx(first, second, target)
    return


def add_not(qc: QuantumCircuit, qubit: Qubit) -> None:
    """
    Add a NOT gate acting on the `qubit` register.
    """
    qc.x(qubit)
    return


def add_xor(qc: QuantumCircuit, first: Qubit, second: Qubit, target: Qubit) -> None:
    """
    Add a XOR gate acting on `first` and `second` and writing the result into `target`.
    """
    qc.cx(first, target)
    qc.cx(second, target)


def add_or(qc: QuantumCircuit, first: Qubit, second: Qubit, target: Qubit) -> None:
    """
    Add a OR gate acting on `first` and `second` and writing the result into `target`.
    """
    qc.cx(first, target)
    qc.cx(second, target)
    qc.ccx(first, second, target)


def add_and_3(qc: QuantumCircuit, qubits: List[Qubit], ancilla: Qubit, target: Qubit) -> None:
    """
    Combine three qubits via an AND operation using one ancilla qubit and saving the result into the `target` qubit.

    :param qc: Underlying QuantumCircuit.
    :param qubits: Qubits to combine via an AND operation.
    :param ancilla: List of 2 ancillary qubits to use for temporary results.
    :param target: If |0> beforehand, this qubit will be set to AND of the other qubits.
    """
    if len(qubits) != 3:
        raise ValueError(f"Expected list of 3 qubits, but got {len(qubits)}.")
    qc.mct(control_qubits=qubits, target_qubit=target, ancilla_qubits=[ancilla], mode='basic')


def add_and_4(qc: QuantumCircuit, qubits: List[Qubit], ancillas: List[Qubit], target: Qubit) -> None:
    """
    Combine four qubits via an AND operation using two ancilla qubits and saving the result into the `target` qubit.

    :param qc: Underlying QuantumCircuit.
    :param qubits: Qubits to combine via an AND operation.
    :param ancillas: List of 2 ancillary qubits to use for temporary results.
    :param target: If |0> beforehand, this qubit will be set to AND of the other qubits.
    """
    if len(qubits) != 4:
        raise ValueError(f"Expected list of 4 qubits, but got {len(qubits)}.")
    if len(ancillas) != 2:
        raise ValueError(f"Expected list of 2 ancilla qubits, but got {len(ancillas)}.")
    qc.mct(control_qubits=qubits, target_qubit=target, ancilla_qubits=ancillas, mode='basic')


def _add_and_4(qc: QuantumCircuit, qubits: List[Qubit], ancillas: List[Qubit], target: Qubit) -> None:
    """
    Combine four qubits via an AND operation using two ancilla qubits and saving the result into the `target` qubit.

    Note: This is worse than the built-in Multi-Toffoli gate, so prefer that.

    :param qc: Underlying QuantumCircuit.
    :param qubits: Qubits to combine via an AND operation.
    :param ancillas: List of 2 ancillary qubits to use for temporary results.
    :param target: If |0> beforehand, this qubit will be set to AND of the other qubits.
    """
    if len(qubits) != 4:
        raise ValueError(f"Expected list of 4 qubits, but got {len(qubits)}.")
    if len(ancillas) != 2:
        raise ValueError(f"Expected list of 2 ancilla qubits, but got {len(ancillas)}.")

    qc.ccx(qubits[0], qubits[1], ancillas[0])
    qc.ccx(qubits[2], qubits[3], ancillas[1])
    qc.ccx(ancillas[0], ancillas[1], target)
    qc.ccx(qubits[2], qubits[3], ancillas[1])
    qc.ccx(qubits[0], qubits[1], ancillas[0])
