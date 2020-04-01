# -*- coding: utf-8 -*-

# This code ist part of quantumcomputation.
#
# Copyright (c) 2020 by DLR.

from typing import *

import pytest
from pytest import approx
from qiskit import *
from qiskit.providers import *
from quantumcomputing.circuits.classic import *


class TestClassicalLogicOperations:

    @pytest.fixture
    def simulator(self) -> BaseBackend:
        return Aer.get_backend("qasm_simulator")

    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        return {'test_runs': 10000,
                'absolute_error': 0.02}

    def test_not(self, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        # Given
        input = QuantumRegister(1, 'input')
        input_measure = ClassicalRegister(1, 'input-measure')
        output_measure = ClassicalRegister(1, 'output-measure')
        qc = QuantumCircuit(input, input_measure, output_measure, name="test-circuit")

        # Mix states to get randomized test cases and record them
        qc.h(input)
        qc.measure(input, input_measure)
        # Add circuit to test
        add_not(qc, input[0])
        # Measure results
        qc.measure(input, output_measure)

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Expected Results are strings 'output input'
        expected_results: Dict[str, float] = {'0 1': 0.5,
                                              '1 0': 0.5}
        assert result == approx(expected_results, abs=config['absolute_error'])

    def test_and(self, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        # Given
        input = QuantumRegister(2, 'input')
        output = QuantumRegister(1, 'output')
        input_measure = ClassicalRegister(2, 'input-measure')
        output_measure = ClassicalRegister(1, 'output-measure')
        qc = QuantumCircuit(input, output, input_measure, output_measure, name="test-circuit")

        # Mix states to get randomized test cases and record them
        qc.h(input)
        qc.measure(input, input_measure)
        # Add circuit to test
        add_and(qc, input[0], input[1], output[0])
        # Measure results
        qc.measure(output, output_measure)

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Expected Results are strings 'output input'
        expected_results: Dict[str, float] = {'0 00': 0.25,
                                              '0 01': 0.25,
                                              '0 10': 0.25,
                                              '1 11': 0.25}
        assert result == approx(expected_results, abs=config['absolute_error'])

    def test_xor(self, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        # Given
        input = QuantumRegister(2, 'input')
        output = QuantumRegister(1, 'output')
        input_measure = ClassicalRegister(2, 'input-measure')
        output_measure = ClassicalRegister(1, 'output-measure')
        qc = QuantumCircuit(input, output, input_measure, output_measure, name="test-circuit")

        # Mix states to get randomized test cases and record them
        qc.h(input)
        qc.measure(input, input_measure)
        # Add circuit to test
        add_xor(qc, input[0], input[1], output[0])
        # Measure results
        qc.measure(output, output_measure)

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Expected Results are strings 'output input'
        expected_results: Dict[str, float] = {'0 00': 0.25,
                                              '1 01': 0.25,
                                              '1 10': 0.25,
                                              '0 11': 0.25}
        assert result == approx(expected_results, abs=config['absolute_error'])

    def test_or(self, simulator: BaseBackend, config: Dict[str, Any]) -> None:
        # Given
        input = QuantumRegister(2, 'input')
        output = QuantumRegister(1, 'output')
        input_measure = ClassicalRegister(2, 'input-measure')
        output_measure = ClassicalRegister(1, 'output-measure')
        qc = QuantumCircuit(input, output, input_measure, output_measure, name="test-circuit")

        # Mix states to get randomized test cases and record them
        qc.h(input)
        qc.measure(input, input_measure)
        # Add circuit to test
        add_or(qc, input[0], input[1], output[0])
        # Measure results
        qc.measure(output, output_measure)

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Expected Results are strings 'output input'
        expected_results: Dict[str, float] = {'0 00': 0.25,
                                              '1 01': 0.25,
                                              '1 10': 0.25,
                                              '1 11': 0.25}
        assert result == approx(expected_results, abs=config['absolute_error'])
