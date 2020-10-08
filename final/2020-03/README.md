# Final Version

This is the final result from the first Quantenhackathon from March 2020.

## Generalities

Due to some weird Python problems on my machine I am not able to
receive the answer from the remote backend and continue
code execution, so I only provide the qasm file for the Grover algorithm
solving the 4-color problem for the 11-vertex graph. However,
when I send the job to IBM, I can download a JSON file from their website showing
that the solution is correct. This means that you need to run it
yourself, which is why I include the `qasm.py` script here.

## Properties

* Solution needs 27 qubits.
* The cost is `60856`.
* Colors are encoded as follows:
  - Red: `00`
  - Blue: `01`
  - Yellow: `10`
  - Green: `11`
* The vertices are added to the quantum circuit in ascending order,
  this means that a result string needs to be read in descending order.
  As an example, the string `10000001111110` means the following:
  - Vertex 1: `10`, i.e. `yellow`
  - Vertex 2: `11`, i.e. `green`
  - Vertex 4: `11`, i.e. `green`
  - Vertex 5: `01`, i.e. `blue`
  - Vertex 6: `00`, i.e. `red`
  - Vertex 8: `00`, i.e. `red`
  - Vertex 9: `10`, i.e. `yellow`

## How to run

### Prerequisites

* At least Python 3.5 
* Qiskit 0.16.2 or 0.18.0 installed in that Python distribution
* Saved IBMQ credentials

You may run `pip install -r requirements.txt` to install all requirements
globally (or in a venv if it is activated). Alternatively, you can run
`python setup.py develop` to install all dependencies and make your
venv aware of all modules.

### Execution

* Run `python quantumcomputing/main.py` to get the file `grover-circuit-4-color.qasm`
  which contains the circuit. This file is already present at this
  location, so you don't need to do this step.
* Run `python qasm.py -d remote -i grover-circuit-4-color.qasm -r 8000`
  to send the job to the IBM simulator. The result will contain lots of
  answers, but there are only 9 entries with a probability _above_ the
  expectation, so they are easy to filter out (well, if your python
  program ever returns from the remote call or you parse the downloaded IBM JSON ...).
  They are:
  - `00011110110001`
  - `00101001110010`
  - `00101001111110`
  - `00101011010010`
  - `00101101110010`
  - `01000010111101`
  - `01001110110001`
  - `10000001111110`
  - `10001101110010`
