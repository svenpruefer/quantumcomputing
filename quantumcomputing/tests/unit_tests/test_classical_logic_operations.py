# -*- coding: utf-8 -*-

# This code ist part of quantumcomputation.
#
# Copyright (c) 2020 by DLR.

from typing import *

import pytest
from pytest import approx
from qiskit import *
from qiskit.providers import *
from quantumcomputing.gates.classic import *


class TestClassicalLogicOperations:

    @pytest.fixture
    def simulator(self) -> BaseBackend:
        return Aer.get_backend("qasm_simulator")

    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        return {'test_runs': 10000,
                'relative_error': 0.05}

    def test_not_on_0(self, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a NOT gate implemented as an X gate::

        |           ┌───┐┌─┐
        |qreg_0: |0>┤ X ├┤M├
        |           └───┘└╥┘
        | creg_0: 0 ══════╩═

        Variable is qreg_0.
        """
        # Given
        qreg = QuantumRegister(1, 'qreg')
        creg = ClassicalRegister(1, 'creg')
        qc = QuantumCircuit(qreg, creg, name="test-circuit")
        add_not(qc, qreg[0])
        qc.measure(qreg[0], creg[0])

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'1': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_not_on_1(self, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a NOT gate implemented as an X gate::

        |           ┌───┐┌─┐
        |qreg_0: |1>┤ X ├┤M├
        |           └───┘└╥┘
        | creg_0: 0 ══════╩═

        Variable is qreg_0.
        """
        # Given
        qreg = QuantumRegister(1, 'qreg')
        creg = ClassicalRegister(1, 'creg')
        qc = QuantumCircuit(qreg, creg, name="test-circuit")
        qc.x(qreg[0])
        add_not(qc, qreg[0])
        qc.measure(qreg[0], creg[0])

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'0': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_and_on_0_0(self, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a AND gate implemented via a Toffoli gate::

        |qreg_0: |0>──■─────
        |             │
        |qreg_1: |0>──■─────
        |           ┌─┴─┐┌─┐
        |qreg_2: |0>┤ X ├┤M├
        |           └───┘└╥┘
        | creg_0: 0 ══════╩═

        Variables are qreg_0 and qreg_1.
        """
        # Given
        qreg = QuantumRegister(3, 'qreg')
        creg = ClassicalRegister(1, 'creg')
        qc = QuantumCircuit(qreg, creg, name="test-circuit-three-qubits")
        add_and(qc, qreg[0], qreg[1], qreg[2])
        qc.measure(qreg[2], creg[0])

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'0': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_and_on_0_1(self, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test an AND gate implemented via a Toffoli gate::

        |qreg_0: |0>──■─────
        |             │
        |qreg_1: |1>──■─────
        |           ┌─┴─┐┌─┐
        |qreg_2: |0>┤ X ├┤M├
        |           └───┘└╥┘
        | creg_0: 0 ══════╩═

        Variables are qreg_0 and qreg_1.
        """
        # Given
        qreg = QuantumRegister(3, 'qreg')
        creg = ClassicalRegister(1, 'creg')
        qc = QuantumCircuit(qreg, creg, name="test-circuit-three-qubits")
        qc.x(qreg[1])
        add_and(qc, qreg[0], qreg[1], qreg[2])
        qc.measure(qreg[2], creg[0])

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'0': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_and_on_1_0(self, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a AND gate implemented via a Toffoli gate::

        |qreg_0: |1>──■─────
        |             │
        |qreg_1: |0>──■─────
        |           ┌─┴─┐┌─┐
        |qreg_2: |0>┤ X ├┤M├
        |           └───┘└╥┘
        | creg_0: 0 ══════╩═

        Variables are qreg_0 and qreg_1.
        """
        # Given
        qreg = QuantumRegister(3, 'qreg')
        creg = ClassicalRegister(1, 'creg')
        qc = QuantumCircuit(qreg, creg, name="test-circuit-three-qubits")
        qc.x(qreg[0])
        add_and(qc, qreg[0], qreg[1], qreg[2])
        qc.measure(qreg[2], creg[0])

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'0': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_and_on_1_1(self, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a AND gate implemented via a Toffoli gate::

        |qreg_0: |1>──■─────
        |             │
        |qreg_1: |1>──■─────
        |           ┌─┴─┐┌─┐
        |qreg_2: |0>┤ X ├┤M├
        |           └───┘└╥┘
        | creg_0: 0 ══════╩═

        Variables are qreg_0 and qreg_1.
        """
        # Given
        qreg = QuantumRegister(3, 'qreg')
        creg = ClassicalRegister(1, 'creg')
        qc = QuantumCircuit(qreg, creg, name="test-circuit-three-qubits")
        qc.x(qreg[0])
        qc.x(qreg[1])
        add_and(qc, qreg[0], qreg[1], qreg[2])
        qc.measure(qreg[2], creg[0])

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'1': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_xor_on_0_0(self, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a XOR gate implemented via a CNOT gate::

        |qreg_0: |0>──■─────
        |           ┌─┴─┐┌─┐
        |qreg_1: |0>┤ X ├┤M├
        |           └───┘└╥┘
        | creg_0: 0 ══════╩═

        Variables are q0_0 and q0_2.
        """
        # Given
        qreg = QuantumRegister(3, 'qreg')
        creg = ClassicalRegister(1, 'creg')
        qc = QuantumCircuit(qreg, creg, name="test-circuit")
        add_xor(qc, qreg[0], qreg[1], qreg[2])
        qc.measure(qreg[2], creg[0])

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'0': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_xor_on_0_1(self, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a XOR gate implemented via a CNOT gate::

        |qreg_0: |0>──■─────
        |           ┌─┴─┐┌─┐
        |qreg_1: |1>┤ X ├┤M├
        |           └───┘└╥┘
        | creg_0: 0 ══════╩═

        Variables are q0_0 and q0_2.
        """
        # Given
        qreg = QuantumRegister(3, 'qreg')
        creg = ClassicalRegister(1, 'creg')
        qc = QuantumCircuit(qreg, creg, name="test-circuit")
        qc.x(qreg[1])
        add_xor(qc, qreg[0], qreg[1], qreg[2])
        qc.measure(qreg[2], creg[0])

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'1': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_xor_on_1_0(self, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a XOR gate implemented via a CNOT gate::

        |qreg_0: |1>──■─────
        |           ┌─┴─┐┌─┐
        |qreg_1: |0>┤ X ├┤M├
        |           └───┘└╥┘
        | creg_0: 0 ══════╩═

        Variables are q0_0 and q0_2.
        """
        # Given
        qreg = QuantumRegister(3, 'qreg')
        creg = ClassicalRegister(1, 'creg')
        qc = QuantumCircuit(qreg, creg, name="test-circuit")
        qc.x(qreg[1])
        add_xor(qc, qreg[0], qreg[1], qreg[2])
        qc.measure(qreg[2], creg[0])

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'1': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_xor_on_1_1(self, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a XOR gate implemented via a CNOT gate::

        |qreg_0: |1>──■─────
        |           ┌─┴─┐┌─┐
        |qreg_1: |1>┤ X ├┤M├
        |           └───┘└╥┘
        | creg_0: 0 ══════╩═

        Variables are q0_0 and q0_2.
        """
        # Given
        qreg = QuantumRegister(3, 'qreg')
        creg = ClassicalRegister(1, 'creg')
        qc = QuantumCircuit(qreg, creg, name="test-circuit")
        qc.x(qreg[0])
        qc.x(qreg[1])
        add_xor(qc, qreg[0], qreg[1], qreg[2])
        qc.measure(qreg[2], creg[0])

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'0': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_or_on_0_0(self, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a OR gate implemented via CNOT and Tofoli gates::

        |qreg_0: |0>──■─────────■─────
        |             │         │  ┌─┐
        |qreg_1: |0>──┼────■────■──┤M├
        |           ┌─┴─┐┌─┴─┐┌─┴─┐└╥┘
        |qreg_2: |0>┤ X ├┤ X ├┤ X ├─╫─
        |           └───┘└───┘└───┘ ║
        | creg_0: 0 ════════════════╩═

        Variables are qreg_0 and qreg_1 and input qreg_2 is assuemd to be |0>.
        """
        # Given
        qreg = QuantumRegister(3, 'qreg')
        creg = ClassicalRegister(1, 'creg')
        qc = QuantumCircuit(qreg, creg, name="test-circuit")
        add_or(qc, qreg[0], qreg[1], qreg[2])
        qc.measure(qreg[2], creg[0])

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'0': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_or_on_0_1(self, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a OR gate implemented via CNOT and Tofoli gates::

        |qreg_0: |0>──■─────────■─────
        |             │         │  ┌─┐
        |qreg_1: |1>──┼────■────■──┤M├
        |           ┌─┴─┐┌─┴─┐┌─┴─┐└╥┘
        |qreg_2: |0>┤ X ├┤ X ├┤ X ├─╫─
        |           └───┘└───┘└───┘ ║
        | creg_0: 0 ════════════════╩═

        Variables are qreg_0 and qreg_1 and input qreg_2 is assuemd to be |0>.
        """
        # Given
        qreg = QuantumRegister(3, 'qreg')
        creg = ClassicalRegister(1, 'creg')
        qc = QuantumCircuit(qreg, creg, name="test-circuit")
        qc.x(qreg[1])
        add_or(qc, qreg[0], qreg[1], qreg[2])
        qc.measure(qreg[2], creg[0])

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'1': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_or_on_1_0(self, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a OR gate implemented via CNOT and Tofoli gates::

        |qreg_0: |1>──■─────────■─────
        |             │         │  ┌─┐
        |qreg_1: |0>──┼────■────■──┤M├
        |           ┌─┴─┐┌─┴─┐┌─┴─┐└╥┘
        |qreg_2: |0>┤ X ├┤ X ├┤ X ├─╫─
        |           └───┘└───┘└───┘ ║
        | creg_0: 0 ════════════════╩═

        Variables are qreg_0 and qreg_1 and input qreg_2 is assumed to be |0>.
        """
        # Given
        qreg = QuantumRegister(3, 'qreg')
        creg = ClassicalRegister(1, 'creg')
        qc = QuantumCircuit(qreg, creg, name="test-circuit")
        qc.x(qreg[0])
        add_or(qc, qreg[0], qreg[1], qreg[2])
        qc.measure(qreg[2], creg[0])

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'1': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_or_on_1_1(self, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a OR gate implemented via CNOT and Tofoli gates::

        |qreg_0: |1>──■─────────■─────
        |             │         │  ┌─┐
        |qreg_1: |1>──┼────■────■──┤M├
        |           ┌─┴─┐┌─┴─┐┌─┴─┐└╥┘
        |qreg_2: |0>┤ X ├┤ X ├┤ X ├─╫─
        |           └───┘└───┘└───┘ ║
        | creg_0: 0 ════════════════╩═

        Variables are qreg_0 and qreg_1 and input qreg_2 is assuemd to be |0>.
        """
        # Given
        qreg = QuantumRegister(3, 'qreg')
        creg = ClassicalRegister(1, 'creg')
        qc = QuantumCircuit(qreg, creg, name="test-circuit")
        qc.x(qreg[0])
        qc.x(qreg[1])
        add_or(qc, qreg[0], qreg[1], qreg[2])
        qc.measure(qreg[2], creg[0])

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'1': 1}
        assert result == approx(expected_results, rel=config['relative_error'])
