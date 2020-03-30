# -*- coding: utf-8 -*-

# This file is part of quantumcomputing.
#
# Copyright (c) 2020 by DLR.

from typing import *

import pytest
from pytest import approx
from qiskit import *
from qiskit.providers import *
from quantumcomputing.gates.adder import add_full_adder


class TestFullAdder:

    @pytest.fixture
    def simulator(self) -> BaseBackend:
        return Aer.get_backend("qasm_simulator")

    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        return {'test_runs': 1000,
                'relative_error': 0.05}

    def test_full_adder_on_0_0_0(self, simulator, config) -> None:
        one_reg = QuantumRegister(1, 'one')
        input_reg = QuantumRegister(3, 'input')
        carry_reg = QuantumRegister(2, 'carry')
        measure = ClassicalRegister(2, 'measure')
        qc = QuantumCircuit(one_reg, input_reg, carry_reg, measure, name="half-adder-circuit")
        qc.x(one_reg[0])
        # Prepare Input
        add_full_adder(qc, input_reg[0], input_reg[1], input_reg[2], one_reg[0], carry_reg[1], carry_reg[0])
        qc.measure(input_reg[2], measure[1])
        qc.measure(carry_reg[0], measure[0])

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'00': 1}
        print(result)
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_full_adder_on_0_0_1(self, simulator, config) -> None:
        one_reg = QuantumRegister(1, 'one')
        input_reg = QuantumRegister(3, 'input')
        carry_reg = QuantumRegister(2, 'carry')
        measure = ClassicalRegister(2, 'measure')
        qc = QuantumCircuit(one_reg, input_reg, carry_reg, measure, name="half-adder-circuit")
        qc.x(one_reg[0])
        # Prepare Input
        qc.x(input_reg[2])
        add_full_adder(qc, input_reg[0], input_reg[1], input_reg[2], one_reg[0], carry_reg[1], carry_reg[0])
        qc.measure(input_reg[2], measure[1])
        qc.measure(carry_reg[0], measure[0])

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'10': 1}
        print(result)
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_full_adder_on_0_1_0(self, simulator, config) -> None:
        one_reg = QuantumRegister(1, 'one')
        input_reg = QuantumRegister(3, 'input')
        carry_reg = QuantumRegister(2, 'carry')
        measure = ClassicalRegister(2, 'measure')
        qc = QuantumCircuit(one_reg, input_reg, carry_reg, measure, name="half-adder-circuit")
        qc.x(one_reg[0])
        # Prepare Input
        qc.x(input_reg[1])
        add_full_adder(qc, input_reg[0], input_reg[1], input_reg[2], one_reg[0], carry_reg[1], carry_reg[0])
        qc.measure(input_reg[2], measure[1])
        qc.measure(carry_reg[0], measure[0])

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'10': 1}
        print(result)
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_full_adder_on_1_0_0(self, simulator, config) -> None:
        one_reg = QuantumRegister(1, 'one')
        input_reg = QuantumRegister(3, 'input')
        carry_reg = QuantumRegister(2, 'carry')
        measure = ClassicalRegister(2, 'measure')
        qc = QuantumCircuit(one_reg, input_reg, carry_reg, measure, name="half-adder-circuit")
        qc.x(one_reg[0])
        # Prepare Input
        qc.x(input_reg[0])
        add_full_adder(qc, input_reg[0], input_reg[1], input_reg[2], one_reg[0], carry_reg[1], carry_reg[0])
        qc.measure(input_reg[2], measure[1])
        qc.measure(carry_reg[0], measure[0])

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'10': 1}
        print(result)
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_full_adder_on_1_1_0(self, simulator, config) -> None:
        one_reg = QuantumRegister(1, 'one')
        input_reg = QuantumRegister(3, 'input')
        carry_reg = QuantumRegister(2, 'carry')
        measure = ClassicalRegister(2, 'measure')
        qc = QuantumCircuit(one_reg, input_reg, carry_reg, measure, name="half-adder-circuit")
        qc.x(one_reg[0])
        # Prepare Input
        qc.x(input_reg[0])
        qc.x(input_reg[1])
        add_full_adder(qc, input_reg[0], input_reg[1], input_reg[2], one_reg[0], carry_reg[1], carry_reg[0])
        qc.measure(input_reg[2], measure[1])
        qc.measure(carry_reg[0], measure[0])

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'01': 1}
        print(result)
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_full_adder_on_1_0_1(self, simulator, config) -> None:
        """
        Tested circuit::

        |             ┌───┐
        |   one_0: |0>┤ X ├───────■─────────■────■────────
        |             ├───┤       │         │    │
        | input_0: |0>┤ X ├───────┼────■────■────┼────────
        |             └───┘       │    │    │    │
        | input_1: |0>───────■────■────┼────┼────┼────────
        |             ┌───┐  │  ┌─┴─┐  │  ┌─┴─┐  │  ┌─┐
        | input_2: |0>┤ X ├──■──┤ X ├──■──┤ X ├──┼──┤M├───
        |             └───┘┌─┴─┐└───┘  │  └───┘┌─┴─┐└╥┘┌─┐
        | carry_0: |0>─────┤ X ├───────┼───────┤ X ├─╫─┤M├
        |                  └───┘     ┌─┴─┐     └─┬─┘ ║ └╥┘
        | carry_1: |0>───────────────┤ X ├───────■───╫──╫─
        |                            └───┘           ║  ║
        |measure_0: 0 ═══════════════════════════════╩══╬═
        |                                               ║
        |measure_1: 0 ══════════════════════════════════╩═
        """
        one_reg = QuantumRegister(1, 'one')
        input_reg = QuantumRegister(3, 'input')
        carry_reg = QuantumRegister(2, 'carry')
        measure = ClassicalRegister(2, 'measure')
        qc = QuantumCircuit(one_reg, input_reg, carry_reg, measure, name="half-adder-circuit")
        qc.x(one_reg[0])
        # Prepare Input
        qc.x(input_reg[0])
        qc.x(input_reg[2])
        add_full_adder(qc, input_reg[0], input_reg[1], input_reg[2], one_reg[0], carry_reg[1], carry_reg[0])
        qc.measure(input_reg[2], measure[1])
        qc.measure(carry_reg[0], measure[0])

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'10': 1}
        print(result)
        print(qc.draw(output="text"))
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_full_adder_on_0_1_1(self, simulator, config) -> None:
        one_reg = QuantumRegister(1, 'one')
        input_reg = QuantumRegister(3, 'input')
        carry_reg = QuantumRegister(2, 'carry')
        measure = ClassicalRegister(2, 'measure')
        qc = QuantumCircuit(one_reg, input_reg, carry_reg, measure, name="half-adder-circuit")
        qc.x(one_reg[0])
        # Prepare Input
        qc.x(input_reg[1])
        qc.x(input_reg[2])
        add_full_adder(qc, input_reg[0], input_reg[1], input_reg[2], one_reg[0], carry_reg[1], carry_reg[0])
        qc.measure(input_reg[2], measure[1])
        qc.measure(carry_reg[0], measure[0])

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'10': 1}
        print(result)
        print(qc.draw(output="text"))
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_full_adder_on_1_1_1(self, simulator, config) -> None:
        one_reg = QuantumRegister(1, 'one')
        input_reg = QuantumRegister(3, 'input')
        carry_reg = QuantumRegister(2, 'carry')
        measure = ClassicalRegister(2, 'measure')
        qc = QuantumCircuit(one_reg, input_reg, carry_reg, measure, name="half-adder-circuit")
        qc.x(one_reg[0])
        # Prepare Input
        qc.x(input_reg[0])
        qc.x(input_reg[1])
        qc.x(input_reg[2])
        add_full_adder(qc, input_reg[0], input_reg[1], input_reg[2], one_reg[0], carry_reg[1], carry_reg[0])
        qc.measure(input_reg[2], measure[1])
        qc.measure(carry_reg[0], measure[0])

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'11': 1}
        print(result)
        assert result == approx(expected_results, rel=config['relative_error'])
