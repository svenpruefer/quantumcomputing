# -*- coding: utf-8 -*-

# This code ist part of quantumcomputation.
#
# Copyright (c) 2020 by DLR.

import math
from typing import *

import pytest
from pytest import approx
from qiskit import *
from qiskit.providers import *


class TestSingleQubitGates:

    @pytest.fixture
    def qc(self) -> QuantumCircuit:
        quantum_register = QuantumRegister(1)
        return QuantumCircuit(quantum_register, name="test-circuit")

    @pytest.fixture
    def simulator(self) -> BaseBackend:
        return Aer.get_backend("qasm_simulator")

    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        return {'test_runs': 1000,
                'relative_error': 0.05}

    def test_identity_on_0(self, qc: QuantumCircuit, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test the following quantum circuit::

        |             ┌────┐ ░ ┌─┐
        |    q0_0: |0>┤ Id ├─░─┤M├
        |             └────┘ ░ └╥┘
        |measure_0: 0 ══════════╩═

        """
        # Given
        qc.iden(0)
        qc.measure_all()

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'0': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_identity_on_superposition(self, qc: QuantumCircuit, simulator: BaseBackend,
                                       config: Dict[str, Any]) -> None:
        """
        Test the following quantum circuit::

        |             ┌─────────────────────────────┐┌────┐ ░ ┌─┐
        |    q0_0: |0>┤ Initialize(0.70711,0.70711) ├┤ Id ├─░─┤M├
        |             └─────────────────────────────┘└────┘ ░ └╥┘
        |measure_0: 0 ═════════════════════════════════════════╩═

        """
        # Given
        qc.initialize([1 / math.sqrt(2), 1 / math.sqrt(2)], qc.qregs)
        qc.iden(0)
        qc.measure_all()

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'0': 0.5, '1': 0.5}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_x_on_0(self, qc: QuantumCircuit, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test the following quantum circuit::

        |             ┌───┐ ░ ┌─┐
        |    q0_0: |0>┤ X ├─░─┤M├
        |             └───┘ ░ └╥┘
        |measure_0: 0 ═════════╩═

        """
        # Given
        qc.x(0)
        qc.measure_all()

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'1': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_x_on_superposition(self, qc: QuantumCircuit, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test the following quantum circuit::

        |             ┌─────────────────────────────┐┌───┐ ░ ┌─┐
        |    q0_0: |0>┤ Initialize(0.70711,0.70711) ├┤ X ├─░─┤M├
        |             └─────────────────────────────┘└───┘ ░ └╥┘
        |measure_0: 0 ════════════════════════════════════════╩═

        """
        # Given
        qc.initialize([1 / math.sqrt(2), 1 / math.sqrt(2)], qc.qregs)
        qc.x(0)
        qc.measure_all()

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'0': 0.5, '1': 0.5}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_y_on_0(self, qc: QuantumCircuit, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test the following quantum circuit::

        |             ┌───┐ ░ ┌─┐
        |    q0_0: |0>┤ Y ├─░─┤M├
        |             └───┘ ░ └╥┘
        |measure_0: 0 ═════════╩═

        """
        # Given
        qc.y(0)
        qc.measure_all()

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'1': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_y_on_superposition(self, qc: QuantumCircuit, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test the following quantum circuit::

                     ┌─────────────────────────────┐┌───┐ ░ ┌─┐
            q0_0: |0>┤ Initialize(0.70711,0.70711) ├┤ Y ├─░─┤M├
                     └─────────────────────────────┘└───┘ ░ └╥┘
        measure_0: 0 ════════════════════════════════════════╩═

        """
        # Given
        qc.initialize([1 / math.sqrt(2), 1 / math.sqrt(2)], qc.qregs)
        qc.x(0)
        qc.measure_all()

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'0': 0.5, '1': 0.5}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_z_on_0(self, qc: QuantumCircuit, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test the following quantum circuit::

        |             ┌───┐ ░ ┌─┐
        |    q0_0: |0>┤ Z ├─░─┤M├
        |             └───┘ ░ └╥┘
        |measure_0: 0 ═════════╩═

        """
        # Given
        qc.z(0)
        qc.measure_all()

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'0': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_z_on_superposition(self, qc: QuantumCircuit, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test the following quantum circuit::

                     ┌─────────────────────────────┐┌───┐ ░ ┌─┐
            q0_0: |0>┤ Initialize(0.70711,0.70711) ├┤ Z ├─░─┤M├
                     └─────────────────────────────┘└───┘ ░ └╥┘
        measure_0: 0 ════════════════════════════════════════╩═

        """
        # Given
        qc.initialize([1 / math.sqrt(2), 1 / math.sqrt(2)], qc.qregs)
        qc.z(0)
        qc.measure_all()

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'0': 0.5, '1': 0.5}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_h_on_0(self, qc: QuantumCircuit, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test the following quantum circuit::

        |             ┌───┐ ░ ┌─┐
        |    q0_0: |0>┤ H ├─░─┤M├
        |             └───┘ ░ └╥┘
        |measure_0: 0 ═════════╩═

        """
        # Given
        qc.h(0)
        qc.measure_all()

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'0': 0.5, '1': 0.5}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_h_on_superposition(self, qc: QuantumCircuit, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test the following quantum circuit::

        |             ┌─────────────────────────────┐┌───┐ ░ ┌─┐
        |    q0_0: |0>┤ Initialize(0.70711,0.70711) ├┤ H ├─░─┤M├
        |             └─────────────────────────────┘└───┘ ░ └╥┘
        |measure_0: 0 ════════════════════════════════════════╩═

        """
        # Given
        qc.initialize([1 / math.sqrt(2), 1 / math.sqrt(2)], qc.qregs)
        qc.h(0)
        qc.measure_all()

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'0': 1}
        assert result == approx(expected_results, rel=config['relative_error'])
