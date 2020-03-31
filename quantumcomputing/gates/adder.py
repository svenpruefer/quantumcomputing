# -*- coding: utf-8 -*-

# This file is part of quantumcomputing.
#
# Copyright (c) 2020 by DLR.

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from quantumcomputing.gates.classic import (add_xor, add_and)


def add_half_adder(qc: QuantumCircuit, a: Qubit, b: Qubit, sum: Qubit, carry: Qubit) -> None:
    add_and(qc, a, b, carry)
    add_xor(qc, a, b, sum)


def add_full_adder(qc: QuantumCircuit, a: Qubit, b: Qubit, c: Qubit, sum_1: Qubit, carry_1: Qubit, sum_2: Qubit,
                   carry_2: Qubit) -> None:
    add_half_adder(qc, b, c, sum_1, carry_1)
    add_half_adder(qc, a, sum_1, sum_2, carry_2)
    qc.cx(carry_1, carry_2)
