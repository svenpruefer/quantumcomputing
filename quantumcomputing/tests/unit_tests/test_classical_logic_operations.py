# -*- coding: utf-8 -*-

# This code ist part of quantumcomputation.
#
# Copyright (c) 2020 by DLR.

from typing import *

import pytest
from pytest import approx
from qiskit import *
from qiskit.providers import *


class TestClassicalLogicOperations:

    @pytest.fixture
    def qc(self) -> QuantumCircuit:
        quantum_register = QuantumRegister(3)
        classical_register = ClassicalRegister(1)
        return QuantumCircuit(quantum_register, classical_register, name="test-circuit")

    @pytest.fixture
    def simulator(self) -> BaseBackend:
        return Aer.get_backend("qasm_simulator")

    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        return {'test_runs': 100000,
                'relative_error': 0.01}

    def test_not_on_0(self, qc: QuantumCircuit, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a NOT gate implemented via Toffoli gates::

        |         ┌──────────────────────────────┐
        |q0_0: |0>┤0                             ├──■─────
        |         │                              │  │
        |q0_1: |0>┤1 Initialize(0,0,0,1,0,0,0,0) ├──■─────
        |         │                              │┌─┴─┐┌─┐
        |q0_2: |0>┤2                             ├┤ X ├┤M├
        |         └──────────────────────────────┘└───┘└╥┘
        |c0_0: 0  ══════════════════════════════════════╩═

        """
        # Given
        qc.initialize([0, 0, 0, 0, 0, 0, 1, 0], qc.qregs)  # Initial state |110>
        qc.ccx(0, 1, 2)
        qc.measure(2, 0)

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'1': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_not_on_1(self, qc: QuantumCircuit, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a NOT gate implemented via Toffoli gates::

        |         ┌──────────────────────────────┐
        |q0_0: |0>┤0                             ├──■─────
        |         │                              │  │
        |q0_1: |0>┤1 Initialize(0,0,0,0,0,0,0,1) ├──■─────
        |         │                              │┌─┴─┐┌─┐
        |q0_2: |0>┤2                             ├┤ X ├┤M├
        |         └──────────────────────────────┘└───┘└╥┘
        |c0_0: 0  ══════════════════════════════════════╩═

        """
        # Given
        qc.initialize([0, 0, 0, 0, 0, 0, 0, 1], qc.qregs)  # Initial state |111>
        qc.ccx(0, 1, 2)
        qc.measure(2, 0)

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'0': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_and_on_0_0(self, qc: QuantumCircuit, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a NOT gate implemented via Toffoli gates::

        |         ┌──────────────────────────────┐
        |q0_0: |0>┤0                             ├──■─────
        |         │                              │  │
        |q0_1: |0>┤1 Initialize(0,0,0,0,0,0,0,0) ├──■─────
        |         │                              │┌─┴─┐┌─┐
        |q0_2: |0>┤2                             ├┤ X ├┤M├
        |         └──────────────────────────────┘└───┘└╥┘
        |c0_0: 0  ══════════════════════════════════════╩═

        """
        # Given
        qc.initialize([1, 0, 0, 0, 0, 0, 0, 0], qc.qregs)  # Initial state |000>
        qc.ccx(0, 1, 2)
        qc.measure(2, 0)

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'0': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_and_on_0_1(self, qc: QuantumCircuit, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a NOT gate implemented via Toffoli gates::

        |         ┌──────────────────────────────┐
        |q3_0: |0>┤0                             ├──■─────
        |         │                              │  │
        |q3_1: |0>┤1 Initialize(0,1,0,0,0,0,0,0) ├──■─────
        |         │                              │┌─┴─┐┌─┐
        |q3_2: |0>┤2                             ├┤ X ├┤M├
        |         └──────────────────────────────┘└───┘└╥┘
        | c3_0: 0 ══════════════════════════════════════╩═

        """
        # Given
        qc.initialize([0, 1, 0, 0, 0, 0, 0, 0], qc.qregs)  # Initial state |001>
        qc.ccx(0, 1, 2)
        qc.measure(2, 0)

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'0': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_and_on_1_0(self, qc: QuantumCircuit, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a NOT gate implemented via Toffoli gates::

        |         ┌──────────────────────────────┐
        |q4_0: |0>┤0                             ├──■─────
        |         │                              │  │
        |q4_1: |0>┤1 Initialize(0,0,1,0,0,0,0,0) ├──■─────
        |         │                              │┌─┴─┐┌─┐
        |q4_2: |0>┤2                             ├┤ X ├┤M├
        |         └──────────────────────────────┘└───┘└╥┘
        | c4_0: 0 ══════════════════════════════════════╩═

        """
        # Given
        qc.initialize([0, 0, 1, 0, 0, 0, 0, 0], qc.qregs)  # Initial state |010>
        qc.ccx(0, 1, 2)
        qc.measure(2, 0)
        # qc.measure_all()

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'0': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_and_on_1_1(self, qc: QuantumCircuit, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a NOT gate implemented via Toffoli gates::

        |         ┌──────────────────────────────┐
        |q0_0: |0>┤0                             ├──■─────
        |         │                              │  │
        |q0_1: |0>┤1 Initialize(0,0,0,1,0,0,0,0) ├──■─────
        |         │                              │┌─┴─┐┌─┐
        |q0_2: |0>┤2                             ├┤ X ├┤M├
        |         └──────────────────────────────┘└───┘└╥┘
        |c0_0: 0  ══════════════════════════════════════╩═

        """
        # Given
        qc.initialize([0, 0, 0, 1, 0, 0, 0, 0], qc.qregs)  # Initial state |011>
        qc.ccx(0, 1, 2)
        qc.measure(2, 0)

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'1': 1}
        assert result == approx(expected_results, rel=config['relative_error'])
