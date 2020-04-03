# This file is part of quantumcomputing.
#
# Copyright (c) 2020 by DLR.

from typing import *

import pytest
from pytest import approx
from qiskit import *
from qiskit.providers import *

from costs.costs import calc_total_costs
from quantumcomputing.circuits.coloring import _compare_internal_edge, _compare_external_edge, VertexColor


class TestFullAdder:

    @pytest.fixture
    def simulator(self) -> BaseBackend:
        return Aer.get_backend("qasm_simulator")

    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        return {'test_runs': 10000,
                'absolute_error': 0.01}

    def test_compare_internal_edge(self, simulator, config) -> None:
        # Given
        first_vertex = QuantumRegister(2, 'first')
        second_vertex = QuantumRegister(2, 'second')
        target = QuantumRegister(1, 'target')
        first_vertex_measure = ClassicalRegister(2, 'first-vertex-measure')
        second_vertex_measure = ClassicalRegister(2, 'second-vertex-measure')
        target_measure = ClassicalRegister(1, 'target-measure')
        qc = QuantumCircuit(first_vertex, second_vertex, target, first_vertex_measure, second_vertex_measure,
                            target_measure,
                            name="test-circuit")

        # Mix states and measure to obtain random test cases
        qc.h(first_vertex)
        qc.h(second_vertex)
        qc.measure(first_vertex, first_vertex_measure)
        qc.measure(second_vertex, second_vertex_measure)
        # Compare edges
        _compare_internal_edge(qc, first_vertex, second_vertex, target[0])
        # Measure result
        qc.measure(target, target_measure)

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        # Expected Results are strings 'first second target'
        expected_results: Dict[str, float] = {'0 00 00': 0.0625,
                                              '0 01 01': 0.0625,
                                              '0 10 10': 0.0625,
                                              '0 11 11': 0.0625,
                                              '1 00 01': 0.0625,
                                              '1 00 10': 0.0625,
                                              '1 00 11': 0.0625,
                                              '1 01 00': 0.0625,
                                              '1 01 10': 0.0625,
                                              '1 01 11': 0.0625,
                                              '1 10 00': 0.0625,
                                              '1 10 01': 0.0625,
                                              '1 10 11': 0.0625,
                                              '1 11 00': 0.0625,
                                              '1 11 01': 0.0625,
                                              '1 11 10': 0.0625}
        assert result == approx(expected_results, abs=config['absolute_error'])
        assert calc_total_costs(qc) - 4 == 129

    def test_compare_external_edge_YELLOW(self, simulator, config) -> None:
        # Given
        vertex = QuantumRegister(2, 'vertex')
        target = QuantumRegister(1, 'target')
        vertex_measure = ClassicalRegister(2, 'vertex-measure')
        target_measure = ClassicalRegister(1, 'target-measure')
        qc = QuantumCircuit(vertex, target, vertex_measure, target_measure, name="test-circuit")

        # Mix states and measure to obtain random test cases
        qc.h(vertex)
        qc.measure(vertex, vertex_measure)
        # Compare edges
        _compare_external_edge(qc, vertex, VertexColor.YELLOW, target[0])
        # Measure result
        qc.measure(target, target_measure)

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        # Expected Results are strings 'first second target'
        expected_results: Dict[str, float] = {'0 10': 0.25, '1 00': 0.25, '1 01': 0.25, '1 11': 0.25}
        assert result == approx(expected_results, abs=config['absolute_error'])
        assert calc_total_costs(qc) - 2 == 72
