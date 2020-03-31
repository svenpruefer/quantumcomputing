# -*- coding: utf-8 -*-

# This file is part of quantumcomputing.
#
# Copyright (c) 2020 by DLR.

from typing import OrderedDict
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller


def calc_u3_cx_gates(qc: QuantumCircuit) -> OrderedDict[str, int]:
    pass_ = Unroller(['u3', 'cx'])
    pm = PassManager(pass_)
    new_circuit = pm.run(qc)
    return new_circuit.count_ops()


def calc_total_costs(qc: QuantumCircuit) -> int:
    counts = calc_u3_cx_gates(qc)
    total = counts['u3'] + 10 * counts['cx']
    return total
