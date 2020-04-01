# -*- coding: utf-8 -*-

# This file is part of quantumcomputing.
#
# Copyright (c) 2020 by DLR.

from typing import *

import pytest
from pytest import approx
from qiskit import *
from qiskit.providers import *

from costs.costs import calc_total_costs
from quantumcomputing.gates.grover import add_grover_without_ancilla_1_0, add_grover_with_ancilla_1_0

class TestCostCalculation:

    def test_cost_grover_without_ancilla(self):
        #Given
        qreg = QuantumRegister(2, "qreg")
        creg = ClassicalRegister(2, "creg")
        qc = QuantumCircuit(qreg, creg, name="grover-without-ancilla-circuit")

        # Add one Grover step
        add_grover_without_ancilla_1_0(qc, qreg)
        # Measure
        qc.measure(qreg[0], creg[0])
        qc.measure(qreg[1], creg[1])

        # When
        result = calc_total_costs(qc)

        # Then
        assert result == 36

    def test_cost_grover_with_ancilla(self):
        #Given
        qreg = QuantumRegister(2, "input")
        ancilla = QuantumRegister(1, "ancilla")
        measure = ClassicalRegister(2, "measure")
        measure_ancilla = ClassicalRegister(1, "ancilla-measure")
        qc = QuantumCircuit(qreg, ancilla, measure, measure_ancilla, name="grover-without-ancilla-circuit")

        # Add one Grover step
        add_grover_with_ancilla_1_0(qc, qreg, ancilla[0])
        # Measure
        qc.measure(qreg[0], measure[0])
        qc.measure(qreg[1], measure[1])

        # When
        result = calc_total_costs(qc)

        # Then
        assert result == 95