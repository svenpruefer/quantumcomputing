# -*- coding: utf-8 -*-

# This file is part of quantumcomputing.
#
# Copyright (c) 2020 by DLR.

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from quantumcomputing.gates.classic import (add_xor, add_and)

def add_half_adder(qc: QuantumCircuit, a: Qubit, b: Qubit, one: Qubit, carry: Qubit) -> None:
    add_and(qc, a, b, carry)
    add_xor(qc, one, a, b)

def add_full_adder(qc: QuantumCircuit, a: Qubit, b: Qubit, c: Qubit, one: Qubit, aux: Qubit, carry: Qubit) -> None:
    add_half_adder(qc, b, c, one, carry)
    add_half_adder(qc, a, c, one, aux)
    add_xor(qc, one, aux, carry)