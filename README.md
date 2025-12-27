# N-Qubit Teleportation Benchmark Suite

This repository contains the companion code for the preprint:

"An N-Qubit Teleportation Protocol with a Generalized GHZ State"
Author: Abdelrahman E. Ahmed

The code implements and benchmarks the proposed N-qubit teleportation protocol. It evaluates performance under:

- ideal simulation
- noisy simulation using device snapshot models
- real quantum hardware (optional)

The implementation uses Qiskit and follows the algorithm described in the paper.

-----------------------------
Features
-----------------------------

This framework provides:

- automatic generation of N-qubit teleportation circuits
- support for dynamic or deferred classical corrections
- ideal Qiskit Aer simulation
- noisy simulation using FakeSherbrooke
- optional execution on IBM Quantum devices
- fidelity estimation using shot-based estimator
- resource usage analysis including:
  * circuit depth
  * total gate count
  * CNOT count
  * number of qubits
  * runtime
- automatic JSON logging
- publication-grade plots

-----------------------------
Repository contents
-----------------------------

n_qubit_teleportation.py
    Main benchmark script

teleportation_benchmark_results.json
    Output results file

teleportation_benchmark_plots.png
    Scalability plots

-----------------------------
Relation to the preprint
-----------------------------

This code reproduces the simulation and resource analysis experiments in the preprint. It is intended for:

- validation of the protocol
- reproducibility of results
- research exploration
- educational demonstration of multi-qubit teleportation

Theoretical background and proofs are in the paper. 
This repository focuses on implementation and benchmarking.

-----------------------------
How to run
-----------------------------

1) Install dependencies

pip install qiskit qiskit-aer qiskit-ibm-runtime matplotlib numpy

2) (Optional) enable IBM Quantum hardware access

Set environment variables:

IBM_QUANTUM_TOKEN=your_token_here
IBM_QUANTUM_INSTANCE=your_instance_string

3) Run benchmark

python n_qubit_teleportation.py

Ideal and noisy simulation run locally.
Hardware execution runs only if credentials are set.

-----------------------------
Output
-----------------------------

The script generates the following files:

teleportation_benchmark_results.json
teleportation_benchmark_plots.png

They contain and visualize:

- fidelity versus N
- gate count versus N
- depth versus N
- CNOT cost versus N
- runtime statistics

-----------------------------
Citation
-----------------------------

If you use this repository, please cite:

An N-Qubit Teleportation Protocol with a Generalized GHZ State
Author: Abdelrahman E. Ahmed
Conference: NILES 2025
Publisher: IEEE
DOI: 10.1109/NILES68063.2025.11232250

(arXiv link will be added after posting)

-----------------------------
License and usage
-----------------------------

This code is provided for research and educational non-commercial use.

Use of real hardware resources is subject to IBM Quantum terms of service.

-----------------------------
Contributions
-----------------------------

Contributions and reproduction of results are welcome.

-----------------------------
Contact
-----------------------------

Abdelrahman E. Ahmed
Alexandria University
Egypt
