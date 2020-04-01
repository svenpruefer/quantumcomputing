# -*- coding: utf-8 -*-

# This file is part of quantumcomputing.
#
# Copyright (c) 2020 by DLR.

from typing import *

import pytest
from pytest import approx
from qiskit import *
from qiskit.providers import *
from quantumcomputing.gates.adder import add_full_adder_7


class TestFullAdder:

    @pytest.fixture
    def simulator(self) -> BaseBackend:
        return Aer.get_backend("qasm_simulator")

    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        return {'test_runs': 10000,
                'absolute_error': 0.01}

    def test_full_adder_7(self, simulator, config) -> None:
        # Given
        input = QuantumRegister(3, 'input')
        aux = QuantumRegister(2, 'aux')
        output = QuantumRegister(2, 'output')
        input_measure = ClassicalRegister(3, 'input-measure')
        output_measure = ClassicalRegister(2, 'output-measure')
        qc = QuantumCircuit(input, aux, output, input_measure, output_measure, name="full-adder-7-circuit")

        # Mix states to get randomized test cases and record them
        qc.h(input)
        qc.measure(input, input_measure)
        # Add full adder
        add_full_adder_7(qc, input[0], input[1], input[2], aux[0], aux[1], output[0], output[1])
        # Measure results
        qc.measure(output, output_measure)

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        # Expected Results are strings 'output input', where output is 'carry' + 'sum'
        expected_results: Dict[str, float] = {'00 000': 0.125,
                                              '01 001': 0.125,
                                              '01 010': 0.125,
                                              '01 100': 0.125,
                                              '10 011': 0.125,
                                              '10 101': 0.125,
                                              '10 110': 0.125,
                                              '11 111': 0.125}
        assert result == approx(expected_results, abs=config['absolute_error'])
