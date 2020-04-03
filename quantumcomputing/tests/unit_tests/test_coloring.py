# This file is part of quantumcomputing.
#
# Copyright (c) 2020 by DLR.

from typing import *

import pytest
from pytest import approx
from qiskit import *
from qiskit.providers import *

from circuits.coloring import _compare_internal_edge, VertexColor, _compare_external_edge, _compare_4_internal_edges, \
    _compare_2_internal_edges
from costs.costs import calc_total_costs


class TestColoringCircuits:

    @pytest.fixture
    def simulator(self) -> BaseBackend:
        return Aer.get_backend("qasm_simulator")

    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        return {'test_runs': 10000,
                'absolute_error': 0.01}

    @pytest.fixture
    def config_slow(self) -> Dict[str, Any]:
        return {'test_runs': 100,
                'absolute_error': 0.1}

    @staticmethod
    def set_color(qc: QuantumCircuit, vertex: QuantumRegister, color: VertexColor) -> None:
        if color == VertexColor.GREEN or color == VertexColor.YELLOW:
            qc.x(vertex[1])
        if color == VertexColor.GREEN or color == VertexColor.BLUE:
            qc.x(vertex[0])

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

    def test_compare_internal_edge_self_reversibility(self, simulator, config) -> None:
        # Given
        first_vertex = QuantumRegister(2, 'first')
        second_vertex = QuantumRegister(2, 'second')
        target = QuantumRegister(1, 'target')
        target_measure = ClassicalRegister(1, 'target-measure')
        qc = QuantumCircuit(first_vertex, second_vertex, target,
                            target_measure,
                            name="test-circuit")

        # Mix states and measure to obtain random test cases
        qc.h(first_vertex)
        qc.h(second_vertex)
        # Compare edges
        _compare_internal_edge(qc, first_vertex, second_vertex, target[0])
        _compare_internal_edge(qc, first_vertex, second_vertex, target[0])
        # Measure result
        qc.measure(target, target_measure)

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'0': 1}
        assert result == approx(expected_results, abs=config['absolute_error'])

    def test_compare_external_edge_yellow(self, simulator, config) -> None:
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

    def test_compare_external_edge_yellow_self_reversibility(self, simulator, config) -> None:
        # Given
        vertex = QuantumRegister(2, 'vertex')
        target = QuantumRegister(1, 'target')
        target_measure = ClassicalRegister(1, 'target-measure')
        qc = QuantumCircuit(vertex, target, target_measure, name="test-circuit")

        # Mix states and measure to obtain random test cases
        qc.h(vertex)
        # Compare edges
        _compare_external_edge(qc, vertex, VertexColor.YELLOW, target[0])
        _compare_external_edge(qc, vertex, VertexColor.YELLOW, target[0])
        # Measure result
        qc.measure(target, target_measure)

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        expected_results: Dict[str, float] = {'0': 1}
        assert result == approx(expected_results, abs=config['absolute_error'])

    def test_compare_4_internal_edges(self, simulator, config_slow) -> None:
        # Given
        edges = [(QuantumRegister(2, f"v_{i}0"), (QuantumRegister(2, f"v_{i}1"))) for i in range(0, 4)]
        ancilla = QuantumRegister(6, 'ancilla')
        target = QuantumRegister(1, 'target')
        first_vertex_measure = ClassicalRegister(2, 'first-vertex-measure')
        second_vertex_measure = ClassicalRegister(2, 'second-vertex-measure')
        target_measure = ClassicalRegister(1, 'target-measure')
        qc = QuantumCircuit(target, ancilla, first_vertex_measure, second_vertex_measure, target_measure,
                            name="test-circuit")
        for v1, v2 in edges:
            qc.add_register(v1, v2)

        # Prepare colors of three of four edges to keep the test size reasonable
        for v1, v2 in edges[1:3]:
            TestColoringCircuits.set_color(qc, v1, VertexColor.RED)
            TestColoringCircuits.set_color(qc, v2, VertexColor.BLUE)
        TestColoringCircuits.set_color(qc, edges[3][0], VertexColor.GREEN)
        TestColoringCircuits.set_color(qc, edges[3][1], VertexColor.YELLOW)
        # Mix the remaining states and measure to obtain random test cases
        qc.h(edges[0][0])
        qc.h(edges[0][1])
        qc.measure(edges[0][0], first_vertex_measure)
        qc.measure(edges[0][1], second_vertex_measure)
        # Compare edges
        _compare_4_internal_edges(qc, edges, list(ancilla), target[0])
        # Measure result
        qc.measure(target, target_measure)

        # When
        job: BaseJob = execute(qc, simulator, shots=config_slow['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config_slow['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        # Expected Results are strings 'target second first'
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
        assert result == approx(expected_results, abs=config_slow['absolute_error'])
        assert calc_total_costs(qc) - 7 == 1247

    def test_compare_2_internal_edges(self, simulator, config) -> None:
        # Given
        edges = [(QuantumRegister(2, f"v_{i}0"), (QuantumRegister(2, f"v_{i}1"))) for i in range(0, 2)]
        ancilla = QuantumRegister(2, 'ancilla')
        target = QuantumRegister(1, 'target')
        edges_measure = [(ClassicalRegister(2, f"v_{i}0-measure"), ClassicalRegister(2, f"v_{i}1-measure")) for i in
                         range(0, 2)]
        target_measure = ClassicalRegister(1, 'target-measure')
        qc = QuantumCircuit(target, ancilla, target_measure, name="test-circuit")
        for v1, v2 in edges:
            qc.add_register(v1, v2)
        for v1m, v2m in edges_measure:
            qc.add_register(v1m, v2m)

        # Mix states and measure to obtain random test cases
        for v in list(sum(edges, ())):
            qc.h(v)
        for (v1, v2), (c1, c2) in zip(edges, edges_measure):
            qc.measure(v1, c1)
            qc.measure(v2, c2)
        # Compare edges
        _compare_2_internal_edges(qc, edges, list(ancilla), target[0])
        # Measure result
        qc.measure(target, target_measure)

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        # Expected Results are strings 'target second first'
        expected_results: Dict[str, float] = {'00 00 00 00 0': 0.0039,
                                              '00 00 00 01 0': 0.0039,
                                              '00 00 00 10 0': 0.0039,
                                              '00 00 00 11 0': 0.0039,
                                              '00 00 01 00 0': 0.0039,
                                              '00 00 01 01 0': 0.0039,
                                              '00 00 01 10 0': 0.0039,
                                              '00 00 01 11 0': 0.0039,
                                              '00 00 10 00 0': 0.0039,
                                              '00 00 10 01 0': 0.0039,
                                              '00 00 10 10 0': 0.0039,
                                              '00 00 10 11 0': 0.0039,
                                              '00 00 11 00 0': 0.0039,
                                              '00 00 11 01 0': 0.0039,
                                              '00 00 11 10 0': 0.0039,
                                              '00 00 11 11 0': 0.0039,
                                              '00 01 00 00 0': 0.0039,
                                              '00 01 00 01 1': 0.0039,
                                              '00 01 00 10 1': 0.0039,
                                              '00 01 00 11 1': 0.0039,
                                              '00 01 01 00 1': 0.0039,
                                              '00 01 01 01 0': 0.0039,
                                              '00 01 01 10 1': 0.0039,
                                              '00 01 01 11 1': 0.0039,
                                              '00 01 10 00 1': 0.0039,
                                              '00 01 10 01 1': 0.0039,
                                              '00 01 10 10 0': 0.0039,
                                              '00 01 10 11 1': 0.0039,
                                              '00 01 11 00 1': 0.0039,
                                              '00 01 11 01 1': 0.0039,
                                              '00 01 11 10 1': 0.0039,
                                              '00 01 11 11 0': 0.0039,
                                              '00 10 00 00 0': 0.0039,
                                              '00 10 00 01 1': 0.0039,
                                              '00 10 00 10 1': 0.0039,
                                              '00 10 00 11 1': 0.0039,
                                              '00 10 01 00 1': 0.0039,
                                              '00 10 01 01 0': 0.0039,
                                              '00 10 01 10 1': 0.0039,
                                              '00 10 01 11 1': 0.0039,
                                              '00 10 10 00 1': 0.0039,
                                              '00 10 10 01 1': 0.0039,
                                              '00 10 10 10 0': 0.0039,
                                              '00 10 10 11 1': 0.0039,
                                              '00 10 11 00 1': 0.0039,
                                              '00 10 11 01 1': 0.0039,
                                              '00 10 11 10 1': 0.0039,
                                              '00 10 11 11 0': 0.0039,
                                              '00 11 00 00 0': 0.0039,
                                              '00 11 00 01 1': 0.0039,
                                              '00 11 00 10 1': 0.0039,
                                              '00 11 00 11 1': 0.0039,
                                              '00 11 01 00 1': 0.0039,
                                              '00 11 01 01 0': 0.0039,
                                              '00 11 01 10 1': 0.0039,
                                              '00 11 01 11 1': 0.0039,
                                              '00 11 10 00 1': 0.0039,
                                              '00 11 10 01 1': 0.0039,
                                              '00 11 10 10 0': 0.0039,
                                              '00 11 10 11 1': 0.0039,
                                              '00 11 11 00 1': 0.0039,
                                              '00 11 11 01 1': 0.0039,
                                              '00 11 11 10 1': 0.0039,
                                              '00 11 11 11 0': 0.0039,
                                              '01 00 00 00 0': 0.0039,
                                              '01 00 00 01 1': 0.0039,
                                              '01 00 00 10 1': 0.0039,
                                              '01 00 00 11 1': 0.0039,
                                              '01 00 01 00 1': 0.0039,
                                              '01 00 01 01 0': 0.0039,
                                              '01 00 01 10 1': 0.0039,
                                              '01 00 01 11 1': 0.0039,
                                              '01 00 10 00 1': 0.0039,
                                              '01 00 10 01 1': 0.0039,
                                              '01 00 10 10 0': 0.0039,
                                              '01 00 10 11 1': 0.0039,
                                              '01 00 11 00 1': 0.0039,
                                              '01 00 11 01 1': 0.0039,
                                              '01 00 11 10 1': 0.0039,
                                              '01 00 11 11 0': 0.0039,
                                              '01 01 00 00 0': 0.0039,
                                              '01 01 00 01 0': 0.0039,
                                              '01 01 00 10 0': 0.0039,
                                              '01 01 00 11 0': 0.0039,
                                              '01 01 01 00 0': 0.0039,
                                              '01 01 01 01 0': 0.0039,
                                              '01 01 01 10 0': 0.0039,
                                              '01 01 01 11 0': 0.0039,
                                              '01 01 10 00 0': 0.0039,
                                              '01 01 10 01 0': 0.0039,
                                              '01 01 10 10 0': 0.0039,
                                              '01 01 10 11 0': 0.0039,
                                              '01 01 11 00 0': 0.0039,
                                              '01 01 11 01 0': 0.0039,
                                              '01 01 11 10 0': 0.0039,
                                              '01 01 11 11 0': 0.0039,
                                              '01 10 00 00 0': 0.0039,
                                              '01 10 00 01 1': 0.0039,
                                              '01 10 00 10 1': 0.0039,
                                              '01 10 00 11 1': 0.0039,
                                              '01 10 01 00 1': 0.0039,
                                              '01 10 01 01 0': 0.0039,
                                              '01 10 01 10 1': 0.0039,
                                              '01 10 01 11 1': 0.0039,
                                              '01 10 10 00 1': 0.0039,
                                              '01 10 10 01 1': 0.0039,
                                              '01 10 10 10 0': 0.0039,
                                              '01 10 10 11 1': 0.0039,
                                              '01 10 11 00 1': 0.0039,
                                              '01 10 11 01 1': 0.0039,
                                              '01 10 11 10 1': 0.0039,
                                              '01 10 11 11 0': 0.0039,
                                              '01 11 00 00 0': 0.0039,
                                              '01 11 00 01 1': 0.0039,
                                              '01 11 00 10 1': 0.0039,
                                              '01 11 00 11 1': 0.0039,
                                              '01 11 01 00 1': 0.0039,
                                              '01 11 01 01 0': 0.0039,
                                              '01 11 01 10 1': 0.0039,
                                              '01 11 01 11 1': 0.0039,
                                              '01 11 10 00 1': 0.0039,
                                              '01 11 10 01 1': 0.0039,
                                              '01 11 10 10 0': 0.0039,
                                              '01 11 10 11 1': 0.0039,
                                              '01 11 11 00 1': 0.0039,
                                              '01 11 11 01 1': 0.0039,
                                              '01 11 11 10 1': 0.0039,
                                              '01 11 11 11 0': 0.0039,
                                              '10 00 00 00 0': 0.0039,
                                              '10 00 00 01 1': 0.0039,
                                              '10 00 00 10 1': 0.0039,
                                              '10 00 00 11 1': 0.0039,
                                              '10 00 01 00 1': 0.0039,
                                              '10 00 01 01 0': 0.0039,
                                              '10 00 01 10 1': 0.0039,
                                              '10 00 01 11 1': 0.0039,
                                              '10 00 10 00 1': 0.0039,
                                              '10 00 10 01 1': 0.0039,
                                              '10 00 10 10 0': 0.0039,
                                              '10 00 10 11 1': 0.0039,
                                              '10 00 11 00 1': 0.0039,
                                              '10 00 11 01 1': 0.0039,
                                              '10 00 11 10 1': 0.0039,
                                              '10 00 11 11 0': 0.0039,
                                              '10 01 00 00 0': 0.0039,
                                              '10 01 00 01 1': 0.0039,
                                              '10 01 00 10 1': 0.0039,
                                              '10 01 00 11 1': 0.0039,
                                              '10 01 01 00 1': 0.0039,
                                              '10 01 01 01 0': 0.0039,
                                              '10 01 01 10 1': 0.0039,
                                              '10 01 01 11 1': 0.0039,
                                              '10 01 10 00 1': 0.0039,
                                              '10 01 10 01 1': 0.0039,
                                              '10 01 10 10 0': 0.0039,
                                              '10 01 10 11 1': 0.0039,
                                              '10 01 11 00 1': 0.0039,
                                              '10 01 11 01 1': 0.0039,
                                              '10 01 11 10 1': 0.0039,
                                              '10 01 11 11 0': 0.0039,
                                              '10 10 00 00 0': 0.0039,
                                              '10 10 00 01 0': 0.0039,
                                              '10 10 00 10 0': 0.0039,
                                              '10 10 00 11 0': 0.0039,
                                              '10 10 01 00 0': 0.0039,
                                              '10 10 01 01 0': 0.0039,
                                              '10 10 01 10 0': 0.0039,
                                              '10 10 01 11 0': 0.0039,
                                              '10 10 10 00 0': 0.0039,
                                              '10 10 10 01 0': 0.0039,
                                              '10 10 10 10 0': 0.0039,
                                              '10 10 10 11 0': 0.0039,
                                              '10 10 11 00 0': 0.0039,
                                              '10 10 11 01 0': 0.0039,
                                              '10 10 11 10 0': 0.0039,
                                              '10 10 11 11 0': 0.0039,
                                              '10 11 00 00 0': 0.0039,
                                              '10 11 00 01 1': 0.0039,
                                              '10 11 00 10 1': 0.0039,
                                              '10 11 00 11 1': 0.0039,
                                              '10 11 01 00 1': 0.0039,
                                              '10 11 01 01 0': 0.0039,
                                              '10 11 01 10 1': 0.0039,
                                              '10 11 01 11 1': 0.0039,
                                              '10 11 10 00 1': 0.0039,
                                              '10 11 10 01 1': 0.0039,
                                              '10 11 10 10 0': 0.0039,
                                              '10 11 10 11 1': 0.0039,
                                              '10 11 11 00 1': 0.0039,
                                              '10 11 11 01 1': 0.0039,
                                              '10 11 11 10 1': 0.0039,
                                              '10 11 11 11 0': 0.0039,
                                              '11 00 00 00 0': 0.0039,
                                              '11 00 00 01 1': 0.0039,
                                              '11 00 00 10 1': 0.0039,
                                              '11 00 00 11 1': 0.0039,
                                              '11 00 01 00 1': 0.0039,
                                              '11 00 01 01 0': 0.0039,
                                              '11 00 01 10 1': 0.0039,
                                              '11 00 01 11 1': 0.0039,
                                              '11 00 10 00 1': 0.0039,
                                              '11 00 10 01 1': 0.0039,
                                              '11 00 10 10 0': 0.0039,
                                              '11 00 10 11 1': 0.0039,
                                              '11 00 11 00 1': 0.0039,
                                              '11 00 11 01 1': 0.0039,
                                              '11 00 11 10 1': 0.0039,
                                              '11 00 11 11 0': 0.0039,
                                              '11 01 00 00 0': 0.0039,
                                              '11 01 00 01 1': 0.0039,
                                              '11 01 00 10 1': 0.0039,
                                              '11 01 00 11 1': 0.0039,
                                              '11 01 01 00 1': 0.0039,
                                              '11 01 01 01 0': 0.0039,
                                              '11 01 01 10 1': 0.0039,
                                              '11 01 01 11 1': 0.0039,
                                              '11 01 10 00 1': 0.0039,
                                              '11 01 10 01 1': 0.0039,
                                              '11 01 10 10 0': 0.0039,
                                              '11 01 10 11 1': 0.0039,
                                              '11 01 11 00 1': 0.0039,
                                              '11 01 11 01 1': 0.0039,
                                              '11 01 11 10 1': 0.0039,
                                              '11 01 11 11 0': 0.0039,
                                              '11 10 00 00 0': 0.0039,
                                              '11 10 00 01 1': 0.0039,
                                              '11 10 00 10 1': 0.0039,
                                              '11 10 00 11 1': 0.0039,
                                              '11 10 01 00 1': 0.0039,
                                              '11 10 01 01 0': 0.0039,
                                              '11 10 01 10 1': 0.0039,
                                              '11 10 01 11 1': 0.0039,
                                              '11 10 10 00 1': 0.0039,
                                              '11 10 10 01 1': 0.0039,
                                              '11 10 10 10 0': 0.0039,
                                              '11 10 10 11 1': 0.0039,
                                              '11 10 11 00 1': 0.0039,
                                              '11 10 11 01 1': 0.0039,
                                              '11 10 11 10 1': 0.0039,
                                              '11 10 11 11 0': 0.0039,
                                              '11 11 00 00 0': 0.0039,
                                              '11 11 00 01 0': 0.0039,
                                              '11 11 00 10 0': 0.0039,
                                              '11 11 00 11 0': 0.0039,
                                              '11 11 01 00 0': 0.0039,
                                              '11 11 01 01 0': 0.0039,
                                              '11 11 01 10 0': 0.0039,
                                              '11 11 01 11 0': 0.0039,
                                              '11 11 10 00 0': 0.0039,
                                              '11 11 10 01 0': 0.0039,
                                              '11 11 10 10 0': 0.0039,
                                              '11 11 10 11 0': 0.0039,
                                              '11 11 11 00 0': 0.0039,
                                              '11 11 11 01 0': 0.0039,
                                              '11 11 11 10 0': 0.0039,
                                              '11 11 11 11 0': 0.0039}
        assert result == approx(expected_results, abs=config['absolute_error'])
        assert calc_total_costs(qc) - 8 == 585
