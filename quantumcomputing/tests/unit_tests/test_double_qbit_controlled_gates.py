# -*- coding: utf-8 -*-

# This code ist part of quantumcomputation.
#
# Copyright (c) 2020 by DLR.

from typing import *

import pytest
from pytest import approx
from qiskit import *
from qiskit.providers import *


class TestDoubleQubitControlledGates:

    @pytest.fixture
    def qc(self) -> QuantumCircuit:
        quantum_register = QuantumRegister(2)
        return QuantumCircuit(quantum_register, name="test-circuit")

    @pytest.fixture
    def simulator(self) -> BaseBackend:
        return Aer.get_backend("qasm_simulator")

    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        return {'test_runs': 100000,
                'relative_error': 0.01}

    def test_bell_state(self, qc: QuantumCircuit, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test the following quantum circuit::

        |                  ┌───┐ ░ ┌─┐
        |    q0_0: |0>─────┤ X ├─░─┤M├───
        |             ┌───┐└─┬─┘ ░ └╥┘┌─┐
        |    q0_1: |0>┤ H ├──■───░──╫─┤M├
        |             └───┘      ░  ║ └╥┘
        |measure_0: 0 ══════════════╩══╬═
        |                              ║
        |measure_1: 0 ═════════════════╩═

        """
        # Given
        qc.h(1)
        qc.cx(1, 0)
        qc.measure_all()

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'00': 0.5, '11': 0.5}
        assert result == approx(expected_results, rel=config['relative_error'])
