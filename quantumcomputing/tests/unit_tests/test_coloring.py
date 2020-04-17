# This file is part of quantumcomputing.
#
# Copyright (c) 2020 by DLR.

from typing import *

import pytest
from pytest import approx
from qiskit import *
from qiskit.providers import *

from quantumcomputing.circuits.coloring import _compare_internal_edge, VertexColor, _compare_external_edge, \
    _compare_4_internal_edges, \
    _compare_2_internal_edges, _compare_4_external_edges, _compare_2_external_edges
from quantumcomputing.costs.costs import calc_total_costs
from quantumcomputing.graph.graph import Graph


class TestColoringCircuits:

    @pytest.fixture
    def simulator(self) -> BaseBackend:
        return Aer.get_backend("qasm_simulator")

    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        return {'test_runs': 10000,
                'test_runs_slow': 10000,
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

    def test_compare_4_external_edges(self, simulator, config) -> None:
        # Given
        edges: List[Tuple[QuantumRegister, VertexColor]] = [
            (QuantumRegister(2, "v1"), VertexColor.YELLOW),  # 10
            (QuantumRegister(2, "v2"), VertexColor.RED),  # 00
            (QuantumRegister(2, "v3"), VertexColor.GREEN),  # 11
            (QuantumRegister(2, "v1"), VertexColor.BLUE)  # 01
        ]
        ancilla = QuantumRegister(6, 'ancilla')
        target = QuantumRegister(1, 'target')
        target_measure = ClassicalRegister(1, 'target-measure')
        input_measure: List[ClassicalRegister] = [
            ClassicalRegister(2, 'v1-measure'),
            ClassicalRegister(2, 'v2-measure'),
            ClassicalRegister(2, 'v3-measure')
        ]
        qc = QuantumCircuit(target, ancilla, target_measure, name="test-circuit")
        for v1, v2 in edges[0:3]:
            qc.add_register(v1)
        for c in input_measure:
            qc.add_register(c)

        # Mix the three independent states and measure to obtain random test cases
        for (v1, v2), c in zip(edges[0:3], input_measure):
            qc.h(v1)
            qc.measure(v1, c)
        # Compare edges
        _compare_4_external_edges(qc, edges, list(ancilla), target[0])
        # Measure result
        qc.measure(target, target_measure)

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        # Expected Results are strings 'v3 v2 v1 target'
        # To interpret the expected results, notice that
        # v1 is next to 10 AND 01 vertex
        # v2 is next to a 00 vertex
        # v3 is next to a 11 vertex
        expected_results: Dict[str, float] = {'00 00 00 0': 0.0156,
                                              '00 00 01 0': 0.0156,
                                              '00 00 10 0': 0.0156,
                                              '00 00 11 0': 0.0156,
                                              '00 01 00 1': 0.0156,
                                              '00 01 01 0': 0.0156,
                                              '00 01 10 0': 0.0156,
                                              '00 01 11 1': 0.0156,
                                              '00 10 00 1': 0.0156,
                                              '00 10 01 0': 0.0156,
                                              '00 10 10 0': 0.0156,
                                              '00 10 11 1': 0.0156,
                                              '00 11 00 1': 0.0156,
                                              '00 11 01 0': 0.0156,
                                              '00 11 10 0': 0.0156,
                                              '00 11 11 1': 0.0156,
                                              '01 00 00 0': 0.0156,
                                              '01 00 01 0': 0.0156,
                                              '01 00 10 0': 0.0156,
                                              '01 00 11 0': 0.0156,
                                              '01 01 00 1': 0.0156,
                                              '01 01 01 0': 0.0156,
                                              '01 01 10 0': 0.0156,
                                              '01 01 11 1': 0.0156,
                                              '01 10 00 1': 0.0156,
                                              '01 10 01 0': 0.0156,
                                              '01 10 10 0': 0.0156,
                                              '01 10 11 1': 0.0156,
                                              '01 11 00 1': 0.0156,
                                              '01 11 01 0': 0.0156,
                                              '01 11 10 0': 0.0156,
                                              '01 11 11 1': 0.0156,
                                              '10 00 00 0': 0.0156,
                                              '10 00 01 0': 0.0156,
                                              '10 00 10 0': 0.0156,
                                              '10 00 11 0': 0.0156,
                                              '10 01 00 1': 0.0156,
                                              '10 01 01 0': 0.0156,
                                              '10 01 10 0': 0.0156,
                                              '10 01 11 1': 0.0156,
                                              '10 10 00 1': 0.0156,
                                              '10 10 01 0': 0.0156,
                                              '10 10 10 0': 0.0156,
                                              '10 10 11 1': 0.0156,
                                              '10 11 00 1': 0.0156,
                                              '10 11 01 0': 0.0156,
                                              '10 11 10 0': 0.0156,
                                              '10 11 11 1': 0.0156,
                                              '11 00 00 0': 0.0156,
                                              '11 00 01 0': 0.0156,
                                              '11 00 10 0': 0.0156,
                                              '11 00 11 0': 0.0156,
                                              '11 01 00 0': 0.0156,
                                              '11 01 01 0': 0.0156,
                                              '11 01 10 0': 0.0156,
                                              '11 01 11 0': 0.0156,
                                              '11 10 00 0': 0.0156,
                                              '11 10 01 0': 0.0156,
                                              '11 10 10 0': 0.0156,
                                              '11 10 11 0': 0.0156,
                                              '11 11 00 0': 0.0156,
                                              '11 11 01 0': 0.0156,
                                              '11 11 10 0': 0.0156,
                                              '11 11 11 0': 0.0156}
        assert result == approx(expected_results, abs=config['absolute_error'])
        assert calc_total_costs(qc) - 6 == 789

    def test_compare_2_external_edges(self, simulator, config) -> None:
        # Given
        edges: List[Tuple[QuantumRegister, VertexColor]] = [
            (QuantumRegister(2, "v1"), VertexColor.YELLOW),  # 10
            (QuantumRegister(2, "v2"), VertexColor.GREEN)  # 11
        ]
        ancilla = QuantumRegister(2, 'ancilla')
        target = QuantumRegister(1, 'target')
        target_measure = ClassicalRegister(1, 'target-measure')
        input_measure: List[ClassicalRegister] = [
            ClassicalRegister(2, 'v1-measure'),
            ClassicalRegister(2, 'v2-measure')
        ]
        qc = QuantumCircuit(target, ancilla, target_measure, name="test-circuit")
        for v1, v2 in edges[0:2]:
            qc.add_register(v1)
        for c in input_measure:
            qc.add_register(c)

        # Mix the two states and measure to obtain random test cases
        for (v1, v2), c in zip(edges[0:2], input_measure):
            qc.h(v1)
            qc.measure(v1, c)
        # Compare edges
        _compare_2_external_edges(qc, edges, list(ancilla), target[0])
        # Measure result
        qc.measure(target, target_measure)

        # When
        job: BaseJob = execute(qc, simulator, shots=config['test_runs'])
        # Calculate relative results
        result: Dict[str, float] = {key: value / config['test_runs'] for key, value in
                                    job.result().get_counts(qc).items()}

        # Then
        # Expected Results are strings 'v2 v1 target'
        # To interpret the expected results, notice that
        # v2 is next to a 11 vertex
        # v1 is next to a 10 vertex
        expected_results: Dict[str, float] = {'00 00 1': 0.0625,
                                              '00 01 1': 0.0625,
                                              '00 10 0': 0.0625,
                                              '00 11 1': 0.0625,
                                              '01 00 1': 0.0625,
                                              '01 01 1': 0.0625,
                                              '01 10 0': 0.0625,
                                              '01 11 1': 0.0625,
                                              '10 00 1': 0.0625,
                                              '10 01 1': 0.0625,
                                              '10 10 0': 0.0625,
                                              '10 11 1': 0.0625,
                                              '11 00 0': 0.0625,
                                              '11 01 0': 0.0625,
                                              '11 10 0': 0.0625,
                                              '11 11 0': 0.0625}
        assert result == approx(expected_results, abs=config['absolute_error'])
        assert calc_total_costs(qc) - 4 == 353

    def test_small_graph_4_coloring(self, simulator, config) -> None:
        # Given
        graph = Graph(
            vertices=['0', '1', '2', '3', '4', '5', '6'],
            edges={
                ('0', '1'),
                ('0', '4'),
                ('0', '5'),
                ('1', '2'),
                ('1', '4'),
                ('1', '5'),
                ('2', '3'),
                ('2', '5'),
                ('2', '6'),
                ('3', '6'),
                ('4', '5'),
                ('5', '6')
            },
            given_colors={
                '0': VertexColor.RED,
                '4': VertexColor.GREEN,
                '5': VertexColor.YELLOW,
                '3': VertexColor.BLUE
            }
        )

        # When
        result: List[Dict[str, VertexColor]] = graph.run_4_cover_grover_algorithm_and_interpret_results(
            simulator,
            config['test_runs'],
            3)

        # Then
        expected_result: List[Dict[str, VertexColor]] = [
            {'1': VertexColor.BLUE, '2': VertexColor.GREEN, '6': VertexColor.RED},
            {'1': VertexColor.BLUE, '2': VertexColor.RED, '6': VertexColor.GREEN}
        ]
        assert len(result) == len(expected_result)
        for x in expected_result:
            assert x in result

    def test_medium_graph_4_coloring(self, simulator, config) -> None:
        # Given
        graph = Graph(
            vertices=['0', '1', '2', '3', '4', '5', '6'],
            edges={
                ('0', '1'),
                ('0', '4'),
                ('0', '5'),
                ('1', '2'),
                ('1', '4'),
                ('1', '5'),
                ('2', '3'),
                ('2', '5'),
                ('2', '6'),
                ('3', '6'),
                ('4', '5'),
                ('5', '6')
            },
            given_colors={
                '0': VertexColor.RED,
                '4': VertexColor.GREEN,
                '3': VertexColor.BLUE
            }
        )

        # When
        result: List[Dict[str, VertexColor]] = graph.run_4_cover_grover_algorithm_and_interpret_results(
            simulator,
            config['test_runs'],
            3)

        # Then
        expected_result: List[Dict[str, VertexColor]] = [
            {'1': VertexColor.BLUE, '2': VertexColor.GREEN, '5': VertexColor.YELLOW, '6': VertexColor.RED},
            {'1': VertexColor.BLUE, '2': VertexColor.RED, '5': VertexColor.YELLOW, '6': VertexColor.GREEN},
            {'1': VertexColor.YELLOW, '2': VertexColor.RED, '5': VertexColor.BLUE, '6': VertexColor.GREEN},
            {'1': VertexColor.YELLOW, '2': VertexColor.RED, '5': VertexColor.BLUE, '6': VertexColor.YELLOW},
            {'1': VertexColor.YELLOW, '2': VertexColor.GREEN, '5': VertexColor.BLUE, '6': VertexColor.RED},
            {'1': VertexColor.YELLOW, '2': VertexColor.GREEN, '5': VertexColor.BLUE, '6': VertexColor.YELLOW}
        ]
        assert len(result) == len(expected_result)
        for x in expected_result:
            assert x in result

    def large_graph_4_coloring(self, simulator, config) -> None:
        """
        DO NOT RUN THIS TEST ON A LOCAL SIMULATOR.

        On SP's machine this takes about 11 minutes for a single (!) run.
        """
        # Given
        graph = Graph(
            vertices=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
            edges={
                ('0', '1'),
                ('0', '4'),
                ('0', '5'),
                ('1', '2'),
                ('1', '4'),
                ('1', '5'),
                ('2', '3'),
                ('2', '5'),
                ('2', '6'),
                ('3', '6'),
                ('4', '5'),
                ('4', '7'),
                ('4', '8'),
                ('4', '9'),
                ('5', '6'),
                ('5', '8'),
                ('5', '9'),
                ('6', '9'),
                ('7', '10'),
                ('8', '9'),
                ('8', '10'),
                ('9', '10')
            },
            given_colors={
                '0': VertexColor.RED,
                '3': VertexColor.BLUE,
                '7': VertexColor.YELLOW,
                '10': VertexColor.GREEN
            }
        )

        # When
        result: Dict[str, int] = graph.run_4_color_grover_algorithm(simulator, 1, 5)

        print(result)
