# -*- coding: utf-8 -*-

# This file is part of quantumcomputing.
#
# Copyright (c) 2020 by DLR.

from typing import *

import pytest
from qiskit import *
from qiskit.providers import *

from gates.grover import add_grover_reflection


class TestGroverReflection:

    @pytest.fixture
    def simulator(self) -> BaseBackend:
        return Aer.get_backend("qasm_simulator")

    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        return {'test_runs': 1000,
                'relative_error': 0.05}

    def test_grover_reflection(self, simulator, config):
        register = QuantumRegister(5, name="input")
        qc = QuantumCircuit(register, name="test-circuit")
        add_grover_reflection(qc, register)

        pass  # TODO Implement test for Grover reflection
