# -*- coding: utf-8 -*-
"""
N-Qubit Teleportation Benchmark Suite

Paper: An N-Qubit Teleportation Protocol with a Generalized GHZ State
Author: Abdelrahman Elsayed Ahmed

This script benchmarks N-qubit teleportation using:
- ideal simulators
- noisy simulators (FakeSherbrooke snapshot)
- real IBM Quantum hardware (optional)

Metrics collected:
- fidelity (shot-based estimator)
- depth
- gate count
- CNOT count
- qubit count
- runtime

Outputs:
- JSON file with benchmark data
- PNG plots
"""

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from qiskit import QuantumCircuit, transpile, ClassicalRegister, QuantumRegister
from qiskit.quantum_info import random_statevector
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke


# -----------------------------
# Configuration
# -----------------------------

IBM_QUANTUM_TOKEN = os.getenv("IBM_QUANTUM_TOKEN", None)
IBM_QUANTUM_INSTANCE = os.getenv("IBM_QUANTUM_INSTANCE", None)
REAL_BACKEND_NAME = os.getenv("IBM_BACKEND", "ibm_fez")

SHOTS = 4000
SEED = 42
N_QUBITS_LIST = [2, 3, 4, 5]


# -----------------------------
# Backend selection
# -----------------------------

def get_backend(mode: str, service: Optional[QiskitRuntimeService] = None):
    if mode == "ideal":
        return AerSimulator()

    if mode == "noisy":
        return AerSimulator.from_backend(FakeSherbrooke())

    if mode == "real":
        if service is None:
            raise ValueError("QiskitRuntimeService required for real execution.")
        return service.backend(REAL_BACKEND_NAME)

    raise ValueError(f"Unknown mode: {mode}")


# -----------------------------
# Circuit construction
# -----------------------------

def build_circuit(
    N: int,
    psi,
    measure_bob_only_for_fidelity: bool = True,
    use_dynamic: bool = True
) -> QuantumCircuit:

    total_qubits = 2 * N + 1
    total_clbits = 2 * N + 1 if measure_bob_only_for_fidelity else N + 1

    qreg = QuantumRegister(total_qubits, "q")
    creg = ClassicalRegister(total_clbits, "c")
    qc = QuantumCircuit(qreg, creg)

    psi_qubits = list(range(N))
    ancilla = N
    bob_qubits = list(range(N + 1, total_qubits))

    # GHZ
    qc.h(ancilla)
    for qb in bob_qubits:
        qc.cx(ancilla, qb)

    # state preparation
    qc.prepare_state(psi.data, psi_qubits)

    # Alice operations
    qc.cx(psi_qubits[0], ancilla)
    for k in range(1, N):
        qc.cx(psi_qubits[k], bob_qubits[k - 1])

    for qb in psi_qubits:
        qc.h(qb)

    # Alice measurements (for dynamic mode)
    if use_dynamic:
        for i in range(N):
            qc.measure(psi_qubits[i], i)
        qc.measure(ancilla, N)

    # Uf
    for k in range(N - 1, 0, -1):
        qc.cx(bob_qubits[k - 1], bob_qubits[k])
        qc.cx(bob_qubits[k], bob_qubits[k - 1])

    # classical corrections
    if use_dynamic:
        with qc.if_test((qc.clbits[N], 1)):
            qc.x(bob_qubits[0])

        for i in range(N):
            with qc.if_test((qc.clbits[i], 1)):
                qc.z(bob_qubits[i])

    else:
        qc.cx(ancilla, bob_qubits[0])
        for i in range(N):
            qc.cz(psi_qubits[i], bob_qubits[i])

    # basis-rotation fidelity trick
    prep_check = QuantumCircuit(len(bob_qubits))
    prep_check.prepare_state(psi.data, range(len(bob_qubits)))
    u_dag = prep_check.inverse().to_gate(label="Uψ†")
    qc.append(u_dag, bob_qubits)

    # measure only Bob for fidelity
    if measure_bob_only_for_fidelity:
        qc.measure(bob_qubits, range(N + 1, total_clbits))

    if not use_dynamic:
        for i in range(N):
            qc.measure(psi_qubits[i], i)
        qc.measure(ancilla, N)

    return qc


# -----------------------------
# Fidelity from measurement counts
# -----------------------------

def fidelity_from_counts(counts: Dict[str, int], N: int) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0

    zeros = "0" * N
    success = 0

    for key, v in counts.items():
        key_clean = key.replace(" ", "")
        bob_bits = key_clean[:N]
        if bob_bits == zeros:
            success += v

    return success / total


# -----------------------------
# Experiment loop
# -----------------------------

def run_experiment(modes: List[str], N_list: List[int], shots: int):
    results = {mode: [] for mode in modes}

    service = None
    if "real" in modes and IBM_QUANTUM_TOKEN:
        QiskitRuntimeService.save_account(
            channel="ibm_cloud",
            token=IBM_QUANTUM_TOKEN,
            instance=IBM_QUANTUM_INSTANCE,
            overwrite=True,
            set_as_default=True
        )
        service = QiskitRuntimeService()

    for N in N_list:
        psi = random_statevector(2 ** N, seed=SEED)

        for mode in modes:

            backend = get_backend(mode, service)
            use_dynamic = (mode == "real")

            qc = build_circuit(N, psi, use_dynamic=use_dynamic)

            depth = qc.depth()
            ops = qc.count_ops()
            cnots = ops.get("cx", 0)
            total_gates = sum(ops.values())

            tqc = transpile(qc, backend)

            start = time.time()

            if mode == "real":
                sampler = Sampler(backend)
                job = sampler.run([tqc], shots=shots)
                result = job.result()
                counts = result[0].data["c"].get_counts()
            else:
                result = backend.run(tqc, shots=shots).result()
                counts = result.get_counts()

            runtime = time.time() - start

            fidelity = fidelity_from_counts(counts, N)

            results[mode].append(
                {
                    "N": N,
                    "fidelity": fidelity,
                    "depth": depth,
                    "gates": int(total_gates),
                    "cnots": int(cnots),
                    "qubits": 2 * N + 1,
                    "runtime": runtime,
                }
            )

            print(f"Mode={mode} N={N} fidelity={fidelity:.3f}")

    return results


# -----------------------------
# Save and plot
# -----------------------------

def save_and_plot(results):

    with open("teleportation_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    modes = list(results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # fidelity
    ax = axes[0]
    for mode in modes:
        X = [d["N"] for d in results[mode]]
        Y = [d["fidelity"] for d in results[mode]]
        ax.plot(X, Y, marker="o", label=mode)
    ax.set_title("Fidelity vs N")
    ax.set_xlabel("N")
    ax.set_ylabel("Fidelity")
    ax.set_ylim(0, 1.05)
    ax.legend()

    # depth
    ax = axes[1]
    for mode in modes:
        X = [d["N"] for d in results[mode]]
        Y = [d["depth"] for d in results[mode]]
        ax.plot(X, Y, marker="s", label=mode)
    ax.set_title("Depth vs N")
    ax.set_xlabel("N")
    ax.set_ylabel("Depth")

    # CNOT count
    ax = axes[2]
    for mode in modes:
        X = [d["N"] for d in results[mode]]
        Y = [d["cnots"] for d in results[mode]]
        ax.plot(X, Y, marker="^", label=mode)
    ax.set_title("CNOT Count vs N")
    ax.set_xlabel("N")
    ax.set_ylabel("CNOTs")

    plt.tight_layout()
    plt.savefig("teleportation_benchmark_plots.png", dpi=300)


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":

    ACTIVE_MODES = ["ideal", "noisy"]  # add "real" if hardware available

    data = run_experiment(ACTIVE_MODES, N_QUBITS_LIST, SHOTS)
    save_and_plot(data)
