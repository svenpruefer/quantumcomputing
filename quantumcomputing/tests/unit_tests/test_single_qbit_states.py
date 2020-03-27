# -*- coding: utf-8 -*-

# This code ist part of quantumcomputation.
#
# Copyright (c) 2020 by DLR.

from typing import *
import pytest
from pytest import approx
from qiskit import *
from qiskit.providers import *


class TestBasicQuantumStates:

    @pytest.fixture
    def qc(self) -> QuantumCircuit:
        quantum_register = QuantumRegister(1)
        return QuantumCircuit(quantum_register, name="test-circuit")

    @pytest.fixture
    def simulator(self) -> BaseBackend:
        return Aer.get_backend("qasm_simulator")

    def test_identity(self, qc: QuantumCircuit, simulator: BaseBackend) -> None:
        # Given
        qc.iden(0)
        qc.measure_all()
        expected_results: Dict[str, int] = {'0': 1000}

        # When
        job: BaseJob = execute(qc, simulator, shots=1000)
        job_result = job.result()
        result: Dict[str, int] = job_result.get_counts(qc)

        # Then
        assert result == approx(expected_results, rel=5e-2)
