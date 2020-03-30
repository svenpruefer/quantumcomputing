# -*- coding: utf-8 -*-

# This file is part of quantumcomputing.
#
# Copyright (c) 2020 by DLR.

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


def add_xor(qc: QuantumCircuit, one: Qubit, first: Qubit, second: Qubit) -> None:
    """
    Add a XOR gate acting on `first` and `second` and writing the result into `second`.
    """
    qc.ccx(one, first, second)


def add_or(qc: QuantumCircuit, first: Qubit, second: Qubit, target: Qubit) -> None:
    """
    Add a OR gate acting on `first` and `second` and writing the result into `target`.
    """
    qc.cx(first, target)
    qc.cx(second, target)
    qc.ccx(first, second, target)
