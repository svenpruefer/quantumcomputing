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
    def qc_triple(self) -> QuantumCircuit:
        quantum_register = QuantumRegister(3)
        classical_register = ClassicalRegister(1)
        return QuantumCircuit(quantum_register, classical_register, name="test-circuit-three-qubits")

    @pytest.fixture
    def qc_quadruple(self) -> QuantumCircuit:
        quantum_register = QuantumRegister(4)
        classical_register = ClassicalRegister(1)
        return QuantumCircuit(quantum_register, classical_register, name="test-circuit-four-qubits")

    @pytest.fixture
    def simulator(self) -> BaseBackend:
        return Aer.get_backend("qasm_simulator")

    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        return {'test_runs': 100000,
                'relative_error': 0.01}

    def test_not_on_0(self, qc_triple, simulator: BaseBackend, config: Dict[str, Any]) -> None:
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

        Variable is q0_0.
        """
        # Given
        qc_triple.initialize([0, 0, 0, 0, 0, 0, 1, 0], qc_triple.qregs)  # Initial state |110>
        qc_triple.ccx(0, 1, 2)
        qc_triple.measure(2, 0)

        # When
        job: BaseJob = execute(qc_triple, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc_triple).items()}

        # Then
        expected_results: Dict[str, float] = {'1': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_not_on_1(self, qc_triple, simulator: BaseBackend, config: Dict[str, Any]) -> None:
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

        Variable is q0_0.
        """
        # Given
        qc_triple.initialize([0, 0, 0, 0, 0, 0, 0, 1], qc_triple.qregs)  # Initial state |111>
        qc_triple.ccx(0, 1, 2)
        qc_triple.measure(2, 0)

        # When
        job: BaseJob = execute(qc_triple, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc_triple).items()}

        # Then
        expected_results: Dict[str, float] = {'0': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_and_on_0_0(self, qc_triple, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a AND gate implemented via Toffoli gates::

        |         ┌──────────────────────────────┐
        |q0_0: |0>┤0                             ├──■─────
        |         │                              │  │
        |q0_1: |0>┤1 Initialize(0,0,0,0,0,0,0,0) ├──■─────
        |         │                              │┌─┴─┐┌─┐
        |q0_2: |0>┤2                             ├┤ X ├┤M├
        |         └──────────────────────────────┘└───┘└╥┘
        |c0_0: 0  ══════════════════════════════════════╩═

        Variables are q0_0 and q0_1.
        """
        # Given
        qc_triple.initialize([1, 0, 0, 0, 0, 0, 0, 0], qc_triple.qregs)  # Initial state |000>
        qc_triple.ccx(0, 1, 2)
        qc_triple.measure(2, 0)

        # When
        job: BaseJob = execute(qc_triple, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc_triple).items()}

        # Then
        expected_results: Dict[str, float] = {'0': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_and_on_0_1(self, qc_triple, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a AND gate implemented via Toffoli gates::

        |         ┌──────────────────────────────┐
        |q3_0: |0>┤0                             ├──■─────
        |         │                              │  │
        |q3_1: |0>┤1 Initialize(0,1,0,0,0,0,0,0) ├──■─────
        |         │                              │┌─┴─┐┌─┐
        |q3_2: |0>┤2                             ├┤ X ├┤M├
        |         └──────────────────────────────┘└───┘└╥┘
        | c3_0: 0 ══════════════════════════════════════╩═

        Variables are q0_0 and q0_1.
        """
        # Given
        qc_triple.initialize([0, 1, 0, 0, 0, 0, 0, 0], qc_triple.qregs)  # Initial state |001>
        qc_triple.ccx(0, 1, 2)
        qc_triple.measure(2, 0)

        # When
        job: BaseJob = execute(qc_triple, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc_triple).items()}

        # Then
        expected_results: Dict[str, float] = {'0': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_and_on_1_0(self, qc_triple, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a AND gate implemented via Toffoli gates::

        |         ┌──────────────────────────────┐
        |q4_0: |0>┤0                             ├──■─────
        |         │                              │  │
        |q4_1: |0>┤1 Initialize(0,0,1,0,0,0,0,0) ├──■─────
        |         │                              │┌─┴─┐┌─┐
        |q4_2: |0>┤2                             ├┤ X ├┤M├
        |         └──────────────────────────────┘└───┘└╥┘
        | c4_0: 0 ══════════════════════════════════════╩═

        Variables are q0_0 and q0_1.
        """
        # Given
        qc_triple.initialize([0, 0, 1, 0, 0, 0, 0, 0], qc_triple.qregs)  # Initial state |010>
        qc_triple.ccx(0, 1, 2)
        qc_triple.measure(2, 0)
        # qc.measure_all()

        # When
        job: BaseJob = execute(qc_triple, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc_triple).items()}

        # Then
        expected_results: Dict[str, float] = {'0': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_and_on_1_1(self, qc_triple, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a AND gate implemented via Toffoli gates::

        |         ┌──────────────────────────────┐
        |q0_0: |0>┤0                             ├──■─────
        |         │                              │  │
        |q0_1: |0>┤1 Initialize(0,0,0,1,0,0,0,0) ├──■─────
        |         │                              │┌─┴─┐┌─┐
        |q0_2: |0>┤2                             ├┤ X ├┤M├
        |         └──────────────────────────────┘└───┘└╥┘
        |c0_0: 0  ══════════════════════════════════════╩═

        Variables are q0_0 and q0_1.
        """
        # Given
        qc_triple.initialize([0, 0, 0, 1, 0, 0, 0, 0], qc_triple.qregs)  # Initial state |011>
        qc_triple.ccx(0, 1, 2)
        qc_triple.measure(2, 0)

        # When
        job: BaseJob = execute(qc_triple, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc_triple).items()}

        # Then
        expected_results: Dict[str, float] = {'1': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_xor_on_0_0(self, qc_triple, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a XOR gate implemented via Toffoli gates::

        |         ┌──────────────────────────────┐
        |q0_0: |0>┤0                             ├──■─────
        |         │                              │  │
        |q0_1: |0>┤1 Initialize(0,0,1,0,0,0,0,0) ├──■─────
        |         │                              │┌─┴─┐┌─┐
        |q0_2: |0>┤2                             ├┤ X ├┤M├
        |         └──────────────────────────────┘└───┘└╥┘
        |c0_0: 0  ══════════════════════════════════════╩═

        Variables are q0_0 and q0_2.
        """
        # Given
        qc_triple.initialize([0, 0, 1, 0, 0, 0, 0, 0], qc_triple.qregs)  # Initial state |010>
        qc_triple.ccx(0, 1, 2)
        qc_triple.measure(2, 0)

        # When
        job: BaseJob = execute(qc_triple, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc_triple).items()}

        # Then
        expected_results: Dict[str, float] = {'0': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_xor_on_0_1(self, qc_triple, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a XOR gate implemented via Toffoli gates::

        |         ┌──────────────────────────────┐
        |q3_0: |0>┤0                             ├──■─────
        |         │                              │  │
        |q3_1: |0>┤1 Initialize(0,0,0,1,0,0,0,0) ├──■─────
        |         │                              │┌─┴─┐┌─┐
        |q3_2: |0>┤2                             ├┤ X ├┤M├
        |         └──────────────────────────────┘└───┘└╥┘
        | c3_0: 0 ══════════════════════════════════════╩═

        Variables are q0_0 and q0_2.
        """
        # Given
        qc_triple.initialize([0, 0, 0, 1, 0, 0, 0, 0], qc_triple.qregs)  # Initial state |011>
        qc_triple.ccx(0, 1, 2)
        qc_triple.measure(2, 0)

        # When
        job: BaseJob = execute(qc_triple, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc_triple).items()}

        # Then
        expected_results: Dict[str, float] = {'1': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_xor_on_1_0(self, qc_triple, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a XOR gate implemented via Toffoli gates::

        |         ┌──────────────────────────────┐
        |q4_0: |0>┤0                             ├──■─────
        |         │                              │  │
        |q4_1: |0>┤1 Initialize(0,0,0,0,0,0,1,0) ├──■─────
        |         │                              │┌─┴─┐┌─┐
        |q4_2: |0>┤2                             ├┤ X ├┤M├
        |         └──────────────────────────────┘└───┘└╥┘
        | c4_0: 0 ══════════════════════════════════════╩═

        Variables are q0_0 and q0_2.
        """
        # Given
        qc_triple.initialize([0, 0, 0, 0, 0, 0, 1, 0], qc_triple.qregs)  # Initial state |110>
        qc_triple.ccx(0, 1, 2)
        qc_triple.measure(2, 0)
        # qc.measure_all()

        # When
        job: BaseJob = execute(qc_triple, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc_triple).items()}

        # Then
        expected_results: Dict[str, float] = {'1': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_xor_on_1_1(self, qc_triple, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a XOR gate implemented via Toffoli gates::

        |         ┌──────────────────────────────┐
        |q0_0: |0>┤0                             ├──■─────
        |         │                              │  │
        |q0_1: |0>┤1 Initialize(0,0,0,0,0,0,0,1) ├──■─────
        |         │                              │┌─┴─┐┌─┐
        |q0_2: |0>┤2                             ├┤ X ├┤M├
        |         └──────────────────────────────┘└───┘└╥┘
        |c0_0: 0  ══════════════════════════════════════╩═

        Variables are q0_0 and q0_2.
        """
        # Given
        qc_triple.initialize([0, 0, 0, 0, 0, 0, 0, 1], qc_triple.qregs)  # Initial state |111>
        qc_triple.ccx(0, 1, 2)
        qc_triple.measure(2, 0)

        # When
        job: BaseJob = execute(qc_triple, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc_triple).items()}

        # Then
        expected_results: Dict[str, float] = {'0': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_or_on_0_0(self, qc_quadruple: QuantumCircuit, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a OR gate implemented via Toffoli gates::

        |         ┌──────────────────────────────────────────────┐
        |q0_0: |0>┤0                                             ├──■─────────■─────
        |         │                                              │  │         │
        |q0_1: |0>┤1                                             ├──■────■────┼─────
        |         │  Initialize(0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0) │  │    │    │
        |q0_2: |0>┤2                                             ├──┼────■────■─────
        |         │                                              │┌─┴─┐┌─┴─┐┌─┴─┐┌─┐
        |q0_3: |0>┤3                                             ├┤ X ├┤ X ├┤ X ├┤M├
        |         └──────────────────────────────────────────────┘└───┘└───┘└───┘└╥┘
        | c0_0: 0 ════════════════════════════════════════════════════════════════╩═

        Variables are q0_0 and q0_1, whereas q0_2 = |1> and q0_3 = |0>.
        """
        # Given
        qc_quadruple.initialize([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                qc_quadruple.qregs)  # Initial state |0100>
        qc_quadruple.ccx(0, 1, 3)
        qc_quadruple.ccx(1, 2, 3)
        qc_quadruple.ccx(0, 2, 3)
        qc_quadruple.measure(3, 0)

        # When
        job: BaseJob = execute(qc_quadruple, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc_quadruple).items()}

        # Then
        expected_results: Dict[str, float] = {'0': 1}
        print(result)
        print(qc_quadruple.draw(output="text"))
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_or_on_0_1(self, qc_quadruple: QuantumCircuit, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a OR gate implemented via Toffoli gates::

        |         ┌──────────────────────────────────────────────┐
        |q0_0: |0>┤0                                             ├──■─────────■─────
        |         │                                              │  │         │
        |q0_1: |0>┤1                                             ├──■────■────┼─────
        |         │  Initialize(0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0) │  │    │    │
        |q0_2: |0>┤2                                             ├──┼────■────■─────
        |         │                                              │┌─┴─┐┌─┴─┐┌─┴─┐┌─┐
        |q0_3: |0>┤3                                             ├┤ X ├┤ X ├┤ X ├┤M├
        |         └──────────────────────────────────────────────┘└───┘└───┘└───┘└╥┘
        | c0_0: 0 ════════════════════════════════════════════════════════════════╩═

        Variables are q0_0 and q0_1, whereas q0_2 = |1> and q0_3 = |0>.
        """
        # Given
        qc_quadruple.initialize([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                qc_quadruple.qregs)  # Initial state |0101>
        qc_quadruple.ccx(0, 1, 3)
        qc_quadruple.ccx(1, 2, 3)
        qc_quadruple.ccx(0, 2, 3)
        qc_quadruple.measure(3, 0)

        # When
        job: BaseJob = execute(qc_quadruple, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc_quadruple).items()}

        # Then
        expected_results: Dict[str, float] = {'1': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_or_on_1_0(self, qc_quadruple: QuantumCircuit, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a OR gate implemented via Toffoli gates::

        |         ┌──────────────────────────────────────────────┐
        |q0_0: |0>┤0                                             ├──■─────────■─────
        |         │                                              │  │         │
        |q0_1: |0>┤1                                             ├──■────■────┼─────
        |         │  Initialize(0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0) │  │    │    │
        |q0_2: |0>┤2                                             ├──┼────■────■─────
        |         │                                              │┌─┴─┐┌─┴─┐┌─┴─┐┌─┐
        |q0_3: |0>┤3                                             ├┤ X ├┤ X ├┤ X ├┤M├
        |         └──────────────────────────────────────────────┘└───┘└───┘└───┘└╥┘
        | c0_0: 0 ════════════════════════════════════════════════════════════════╩═

        Variables are q0_0 and q0_1, whereas q0_2 = |1> and q0_3 = |0>.
        """
        # Given
        qc_quadruple.initialize([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                qc_quadruple.qregs)  # Initial state |0110>
        qc_quadruple.ccx(0, 1, 3)
        qc_quadruple.ccx(1, 2, 3)
        qc_quadruple.ccx(0, 2, 3)
        qc_quadruple.measure(3, 0)

        # When
        job: BaseJob = execute(qc_quadruple, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc_quadruple).items()}

        # Then
        expected_results: Dict[str, float] = {'1': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_or_on_1_1(self, qc_quadruple: QuantumCircuit, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        """
        Test a OR gate implemented via Toffoli gates::

        |         ┌──────────────────────────────────────────────┐
        |q0_0: |0>┤0                                             ├──■─────────■─────
        |         │                                              │  │         │
        |q0_1: |0>┤1                                             ├──■────■────┼─────
        |         │  Initialize(0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0) │  │    │    │
        |q0_2: |0>┤2                                             ├──┼────■────■─────
        |         │                                              │┌─┴─┐┌─┴─┐┌─┴─┐┌─┐
        |q0_3: |0>┤3                                             ├┤ X ├┤ X ├┤ X ├┤M├
        |         └──────────────────────────────────────────────┘└───┘└───┘└───┘└╥┘
        | c0_0: 0 ════════════════════════════════════════════════════════════════╩═

        Variables are q0_0 and q0_1, whereas q0_2 = |1> and q0_3 = |0>.
        """
        # Given
        qc_quadruple.initialize([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                qc_quadruple.qregs)  # Initial state |0111>
        qc_quadruple.ccx(0, 1, 3)
        qc_quadruple.ccx(1, 2, 3)
        qc_quadruple.ccx(0, 2, 3)
        qc_quadruple.measure(3, 0)

        # When
        job: BaseJob = execute(qc_quadruple, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc_quadruple).items()}

        # Then
        expected_results: Dict[str, float] = {'1': 1}
        assert result == approx(expected_results, rel=config['relative_error'])
