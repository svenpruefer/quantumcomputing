# -*- coding: utf-8 -*-

# This file is part of qsoc.
#
# Copyright (c) 2020 by DLR.
from typing import List, Set

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit


def _add_2_bit_oracle_without_ancilla_1_0(qc: QuantumCircuit, register: QuantumRegister) -> None:
    """
    Add a Grover oracle for the state |10> on a 2-qubit QuantumRegister.
    When acting on a completely mixed state in this register, this will flip the amplitude of the state |10>.
    The circuit will look like the following::

    | TODO

    :param qc: Underlying QuantumCircuit.
    :param register: QuantumRegister containing exactly two qubits with the qubits ordered in the usual ascending order.
    """
    if len(list(register)) != 2:
        raise ValueError(f"Need QuantumRegister with exactly 2 qubits, but got {len(list(register))} instead.")

    qc.x(register[0])
    qc.cz(register[1], register[0])
    qc.x(register[0])


def _add_grover_step_without_ancilla_1_0(qc: QuantumCircuit, register: QuantumRegister) -> None:
    """
    Add a Grover algorithm step finding the state |10> on a given QuantumRegister with two qubits.
    The circuit will look like the following::

    | TODO

    :param qc: Underlying QuantumCircuit.
    :param register: QuantumRegister containing exactly two qubits with the qubits orderd in the usual ascending order.
    :return:
    """
    if len(list(register)) != 2:
        raise ValueError(f"Need QuantumRegister with exactly 2 qubits, but got {len(list(register))} instead.")

    _add_2_bit_oracle_without_ancilla_1_0(qc, register)  # Oracle
    add_grover_reflection_no_ancilla(qc, register)  # Reflection


def add_grover_without_ancilla_1_0(qc: QuantumCircuit, register: QuantumRegister) -> None:
    """
    Add a Grover algorithm finding the state |10> on a given QuantumRegister with two qubits.
    The circuit will look like the following::

    | TODO

    :param qc: Underlying QuantumCircuit.
    :param register: QuantumRegister containing exactly two qubits with the qubits orderd in the usual ascending order.
    """
    if len(list(register)) != 2:
        raise ValueError(f"Need QuantumRegister with exactly 2 qubits, but got {len(list(register))} instead.")

    qc.h(register)  # Mix states initially
    _add_grover_step_without_ancilla_1_0(qc, register)  # Repeat Grover iteration just once


def add_grover_with_ancilla_1_0(qc: QuantumCircuit, register: QuantumRegister, ancilla: Qubit) -> None:
    """
    Add a Grover algorithm finding the state |10> on a given QuantumRegister with two qubits using a single ancillary qubit.
    The circuit will look like the following::

    | TODO

    :param qc: Underlying QuantumCircuit.
    :param register: QuantumRegister containing exactly two qubits with the qubits orderd in the usual ascending order.
    :param ancilla: Ancillary Qubit.
    """
    if len(list(register)) != 2:
        raise ValueError(f"Need QuantumRegister with exactly 2 qubits, but got {len(list(register))} instead.")

    # Mark states with a 1 at a later determined position by flipping their amplitude
    qc.x(ancilla)
    qc.h(ancilla)
    # Mix input states
    qc.h(register)
    # Execute Grover iteration just once
    _add_grover_step_with_ancilla_1_0(qc, register, ancilla)


def _add_grover_step_with_ancilla_1_0(qc: QuantumCircuit, register: QuantumRegister, ancilla: Qubit) -> None:
    """
    Add a Grover algorithm step finding the state |10> on a given QuantumRegister with two qubits.
    The circuit will look like the following::

    | TODO

    :param qc: Underlying QuantumCircuit.
    :param register: QuantumRegister containing exactly two qubits with the qubits orderd in the usual ascending order.
    :return:
    """
    if len(list(register)) != 2:
        raise ValueError(f"Need QuantumRegister with exactly 2 qubits, but got {len(list(register))} instead.")

    _add_2_bit_oracle_with_ancilla_1_0(qc, register, ancilla)  # Oracle
    add_grover_reflection_no_ancilla(qc, register)  # Reflection


def _add_2_bit_oracle_with_ancilla_1_0(qc: QuantumCircuit, register: QuantumRegister, ancilla: Qubit) -> None:
    """
    Add a Grover oracle for the state |10> on a 2-qubit QuantumRegister using a 1-qubit ancilla.
    When acting on a completely mixed state in this register, this will flip the amplitude of the state |10>.
    The circuit will look like the following::

    | TODO

    :param qc: Underlying QuantumCircuit.
    :param register: QuantumRegister containing exactly two qubits with the qubits ordered in the usual ascending order.
    :param ancilla: Ancillary qubit.
    """
    if len(list(register)) != 2:
        raise ValueError(f"Need QuantumRegister with exactly 2 qubits, but got {len(list(register))} instead.")

    qc.x(register[0])
    qc.ccx(register[1], register[0], ancilla)
    qc.x(register[0])


def add_grover_reflection_no_ancilla(qc: QuantumCircuit, register: QuantumRegister) -> None:
    """
    Add a Grover reflection (or amplitude amplification) acting on a specified QuantumRegister in a QuantumCircuit
    without the use of an ancilla qubit.

    The created circuit will look like the following::

    |            ┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐
    |input_0: |0>┤ H ├┤ X ├┤ H ├┤ X ├┤ H ├┤ X ├┤ H ├
    |            ├───┤├───┤└───┘└─┬─┘└───┘├───┤├───┤
    |input_1: |0>┤ H ├┤ X ├───────■───────┤ X ├┤ H ├
    |            ├───┤├───┤       │       ├───┤├───┤
    |input_2: |0>┤ H ├┤ X ├───────■───────┤ X ├┤ H ├
    |            ├───┤├───┤       │       ├───┤├───┤
    |input_3: |0>┤ H ├┤ X ├───────■───────┤ X ├┤ H ├
    |            ├───┤├───┤       │       ├───┤├───┤
    |input_4: |0>┤ H ├┤ X ├───────■───────┤ X ├┤ H ├
    |            └───┘└───┘               └───┘└───┘

    Beware that not using an ancilla qubit requires lots of additional quantum gates!

    :param qc: Underlying QuantumCircuit.
    :param register: QuantumRegister containing the qubits whose amplitudes shall be amplified.
    """
    qc.h(register)
    qc.x(register)
    qc.h(register[0])
    qc.mct(q_controls=register[1:], q_target=register[0], q_ancilla=None, mode='noancilla')
    qc.h(register[0])
    qc.x(register)
    qc.h(register)


def add_grover_reflection_with_ancilla(qc: QuantumCircuit, register: QuantumRegister, ancillas: List[Qubit]) -> None:
    """
    Add a Grover reflection (or amplitude amplification) acting on a specified QuantumRegister in a QuantumCircuit
    using ancilla qubits.

    Notice that you need `#{qubits in register} - 3` ancilla qubits. This is because this implementation uses a
    Multi-Toffoli-Gate with one target qubit and the basic qiskit implementation needs `#{control qubits} - 2`
    ancillary qubits.

    :param qc: Underlying QuantumCircuit.
    :param register: QuantumRegister containing the qubits whose amplitudes shall be amplified.
    :param ancillas: List of ancilla qubits to use.
    """
    if len(ancillas) < len(list(register)) - 3:
        raise ValueError(f"Need {len(list(register)) - 3} many ancilla qubits but got {len(ancillas)}")
    qc.h(register)
    qc.x(register)
    qc.h(register[0])
    qc.mct(q_controls=register[1:], q_target=register[0], q_ancilla=ancillas)
    qc.h(register[0])
    qc.x(register)
    qc.h(register)


def add_grover_reflection_with_ancilla_on_registers(qc: QuantumCircuit, registers: Set[QuantumRegister],
                                       ancillas: Set[Qubit]) -> None:
    """
    Add a Grover reflection (or amplitude amplification) acting on all qubits in a set of QuantumRegisters
    in a QuantumCircuit using ancilla qubits.

    Notice that you need `#{qubits in all registers} - 3` ancilla qubits. This is because this implementation uses a
    Multi-Toffoli-Gate with one target qubit and the basic qiskit implementation needs `#{control qubits} - 2`
    ancillary qubits.

    :param qc: Underlying QuantumCircuit.
    :param registers: Set of QuantumRegister containing the qubits whose amplitudes shall be amplified.
    :param ancillas: Set of ancilla qubits to use.
    """
    qubits: List[Qubit] = [qubit for register in registers for qubit in list(register)]
    if len(ancillas) < len(qubits) - 3:
        raise ValueError(f"Need {len(qubits) - 3} many ancilla qubits but got {len(ancillas)}")
    for qubit in qubits:
        qc.h(qubit)
        qc.x(qubit)
    qc.h(qubits[0])
    qc.mct(q_controls=qubits[1:], q_target=qubits[0], q_ancilla=list(ancillas))
    qc.h(qubits[0])
    for qubit in qubits:
        qc.x(qubit)
        qc.h(qubit)
