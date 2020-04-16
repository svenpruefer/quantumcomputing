# -*- coding: utf-8 -*-

# This file is part of quantumcomputing.
#
# Copyright (c) 2020 by DLR.

from typing import *

import pytest
from pytest import approx
from qiskit import *
from qiskit.providers import *
from quantumcomputing.circuits.grover import add_grover_without_ancilla_1_0, add_grover_with_ancilla_1_0


class TestGrover:

    @pytest.fixture
    def simulator(self) -> BaseBackend:
        return Aer.get_backend("qasm_simulator")

    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        return {'test_runs': 1000,
                'relative_error': 0.05}

    def test_grover_without_ancilla_on_1_0(self, simulator, config):
        """
        This tests the following circuit::

        |           ┌───┐┌───┐   ┌───┐ ░ ┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐ ░ ┌─┐
        |qreg_0: |0>┤ H ├┤ X ├─■─┤ X ├─░─┤ H ├┤ X ├┤ H ├┤ X ├┤ H ├┤ X ├┤ H ├─░─┤M├──────
        |           ├───┤└───┘ │ └───┘ ░ ├───┤├───┤└───┘└─┬─┘├───┤└───┘├───┤ ░ └╥┘   ┌─┐
        |qreg_1: |0>┤ H ├──────■───────░─┤ H ├┤ X ├───────■──┤ X ├─────┤ H ├─░──╫────┤M├
        |           └───┘              ░ └───┘└───┘          └───┘     └───┘ ░  ║    └╥┘
        | creg_0: 0 ══════════════════════════════════════════════════════════════════╩═
        |                                                                             ║
        | creg_1: 0 ══════════════════════════════════════════════════════════════════╩═

        """
        input = QuantumRegister(2, "input")
        measure = ClassicalRegister(2, "measure")
        qc = QuantumCircuit(input, measure, name="grover-without-ancilla-circuit")

        # Add Grover algorithm
        add_grover_without_ancilla_1_0(qc, input)

        # Measure
        qc.measure(input, measure)

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'10': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_grover_with_ancilla_on_1_0(self, simulator, config):
        input = QuantumRegister(2, "input")
        ancilla = QuantumRegister(1, "ancilla")
        measure = ClassicalRegister(2, "measure")
        measure_ancilla = ClassicalRegister(1, "ancilla-measure")
        qc = QuantumCircuit(input, ancilla, measure, measure_ancilla, name="grover-without-ancilla-circuit")

        # Add Grover circuit
        add_grover_with_ancilla_1_0(qc, input, ancilla[0])
        # Measure
        qc.measure(input, measure)
        qc.measure(ancilla, measure_ancilla)

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'0 10': 0.5, '1 10': 0.5}
        assert result == approx(expected_results, rel=config['relative_error'])
