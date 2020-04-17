#!/usr/bin/python3 -W ignore
#!/opt/local/bin/python -W ignore
# path can be different

import argparse
import json
import qiskit
import sys

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-d', '--device',
                    help='type of backend (default print), local, remote, state, unit',
                    type=str, default='print')
parser.add_argument('-i', '--input',
                    help='input file (default stdin)',
                    type=str, default='')
parser.add_argument('-r', '--runs',
                    help='number of runs (default 1000, max 8192)',
                    type=int, default=1000)
args = parser.parse_args()

hub = 'ibm-q-unibw'
group = 'training'
project = 'challenge'
fac = 10   # factor for CX gates over U3 gates

def connect(hub = 'ibm-q', group = 'open', project = 'main'):
    qiskit.IBMQ.load_account()
    provider = qiskit.IBMQ.get_provider(hub, group, project)
    return provider

if (args.device == 'local'): # local simulator
    backend = qiskit.Aer.get_backend('qasm_simulator')
elif (args.device == 'remote'): # remote simulator
    provider = connect(hub, group, project)
    backend = provider.get_backend('ibmq_qasm_simulator')
elif (args.device == 'state'): # statevector
    backend = qiskit.Aer.get_backend('statevector_simulator')
elif (args.device == 'unit'): # unitary
    backend = qiskit.Aer.get_backend('unitary_simulator')
elif (args.device == 'print'): # printer
    backend = ""
else:
    print("no such backend")
    exit(1)

if (args.input == ''):
    inp = sys.stdin.readlines()
else:
    inp = open(args.input, 'r').readlines()

if (args.device == "unit"):
    inp = list(filter(lambda str: not('measure' in str), inp))
inp = "\n".join(inp)

### begin of circuit construction
if (inp != ""):
    qc = qiskit.QuantumCircuit.from_qasm_str(inp)
### end of circuit construction

if (inp == "" or args.runs == 0):
    provider = connect(hub, group, project)
    backend = provider.get_backend('ibmq_qasm_simulator')
    from qiskit.tools.monitor import backend_monitor
    backend_monitor(backend)
    out = backend.configuration()
    print("basegates:", file=sys.stderr, end=' ')
    print(out.basis_gates, file=sys.stderr)
    print("couplings:", file=sys.stderr, end=' ')
    print(out.coupling_map, file=sys.stderr)
elif (args.device == "print"):
    print(qc, file=sys.stdout)
    print(qc.qasm(), file=sys.stderr)
elif (args.device == "state"):
    job = qiskit.execute(qc, backend)
    res = job.result()
    out = res.get_statevector()
    print(out, file=sys.stdout)
    pas = qiskit.transpiler.passes.Unroller(['u3', 'cx'])
    pam = qiskit.transpiler.PassManager(pas)
    new = pam.run(qc)
    num = dict(new.count_ops())
    u3c = num["u3"]
    cxc = num["cx"]
    print("U3     : " + str(u3c), file=sys.stderr)
    print("CX     : " + str(cxc), file=sys.stderr)
    print("weight : " + str(u3c + fac * cxc), file=sys.stderr)
    print("depth  : " + str(new.depth()), file=sys.stderr)
    print("size   : " + str(new.size()), file=sys.stderr)
    print("width  : " + str(new.width()), file=sys.stderr)
    print(new, file=sys.stderr)
elif (args.device == "unit"):
    job = qiskit.execute(qc, backend)
    res = job.result()
    out = res.get_unitary()
    print(out, file=sys.stdout)
    print(list(map(lambda x:x[0], out)), file=sys.stderr)
else:
    job = qiskit.execute(qc, backend, shots=args.runs)
    res = job.result()
    out = res.get_counts()
    print(json.dumps(out, indent=2, sort_keys=True), file=sys.stdout)
    print(out, file=sys.stderr)
