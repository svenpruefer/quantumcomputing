# This file is part of qsoc.
#
# Copyright (c) 2020 by DLR.

#from qiskit import IBMQ

from costs.costs import calc_total_costs
from qsoc.circuits.coloring import VertexColor
from qsoc.graph.graph import Graph


def main():
    """
    Main entry point for qsoc.
    """
    # Get Backend
    #IBMQ.load_account()
    #unibw = IBMQ.get_provider(hub='ibm-q-unibw', group='training', project='challenge')
    #backend = unibw.get_backend('ibmq_qasm_simulator')

    # Define Graph
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

    # Run Grover
    qc = graph.get_4_color_grover_algorithm_with_measurements(5)

    print(f"Costs: {calc_total_costs(qc)}")

    with open("grover-circuit-4-color.qasm", "w") as file:
        file.write(qc.qasm())


if __name__ == "__main__":
    # execute only if run as a script
    main()
