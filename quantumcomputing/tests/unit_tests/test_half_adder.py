# -*- coding: utf-8 -*-

# This file is part of quantumcomputing.
#
# Copyright (c) 2020 by DLR.

from typing import *

import pytest
from pytest import approx
from qiskit import *
from qiskit.providers import *
from quantumcomputing.gates.adder import add_half_adder


class TestHalfAdder:

    @pytest.fixture
    def simulator(self) -> BaseBackend:
        return Aer.get_backend("qasm_simulator")

    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        return {'test_runs': 1000,
                'relative_error': 0.05}

    def test_half_adder_on_0_0(self, simulator, config) -> None:
        input_reg = QuantumRegister(3, 'input')
        carry_reg = QuantumRegister(1, 'carry')
        sum_reg = QuantumRegister(1, 'sum')
        measure = ClassicalRegister(2, 'measure')
        qc = QuantumCircuit(input_reg, carry_reg, sum_reg, measure, name="half-adder-circuit")
        # Prepare Input
        add_half_adder(qc, input_reg[0], input_reg[1], sum_reg[0], carry_reg[0])
        qc.measure(sum_reg[0], measure[1])
        qc.measure(carry_reg[0], measure[0])

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'00': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_half_adder_on_0_1(self, simulator, config) -> None:
        input_reg = QuantumRegister(3, 'input')
        carry_reg = QuantumRegister(1, 'carry')
        sum_reg = QuantumRegister(1, 'sum')
        measure = ClassicalRegister(2, 'measure')
        qc = QuantumCircuit(input_reg, carry_reg, sum_reg, measure, name="half-adder-circuit")
        # Prepare Input
        qc.x(input_reg[1])
        add_half_adder(qc, input_reg[0], input_reg[1], sum_reg[0], carry_reg[0])
        qc.measure(sum_reg[0], measure[1])
        qc.measure(carry_reg[0], measure[0])

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'10': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_half_adder_on_1_0(self, simulator, config) -> None:
        input_reg = QuantumRegister(3, 'input')
        carry_reg = QuantumRegister(1, 'carry')
        sum_reg = QuantumRegister(1, 'sum')
        measure = ClassicalRegister(2, 'measure')
        qc = QuantumCircuit(input_reg, carry_reg, sum_reg, measure, name="half-adder-circuit")
        # Prepare Input
        qc.x(input_reg[0])
        add_half_adder(qc, input_reg[0], input_reg[1], sum_reg[0], carry_reg[0])
        qc.measure(sum_reg[0], measure[1])
        qc.measure(carry_reg[0], measure[0])

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'10': 1}
        assert result == approx(expected_results, rel=config['relative_error'])

    def test_half_adder_on_1_1(self, simulator, config) -> None:
        input_reg = QuantumRegister(3, 'input')
        carry_reg = QuantumRegister(1, 'carry')
        sum_reg = QuantumRegister(1, 'sum')
        measure = ClassicalRegister(2, 'measure')
        qc = QuantumCircuit(input_reg, carry_reg, sum_reg, measure, name="half-adder-circuit")
        # Prepare Input
        qc.x(input_reg[0])
        qc.x(input_reg[1])
        add_half_adder(qc, input_reg[0], input_reg[1], sum_reg[0], carry_reg[0])
        qc.measure(sum_reg[0], measure[1])
        qc.measure(carry_reg[0], measure[0])

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'01': 1}
        assert result == approx(expected_results, rel=config['relative_error'])
