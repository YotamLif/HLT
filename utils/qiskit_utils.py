import itertools
from typing import List, Optional, Union

import numpy as np
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.providers import Backend
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.library import SetDensityMatrix, SaveDensityMatrix
from qiskit.quantum_info import Pauli


def get_density_matrix_circuit(sys_size: int, density_matrix: np.array) -> QuantumCircuit:
    circ = QuantumCircuit(sys_size)
    circ.append(SetDensityMatrix(density_matrix), range(sys_size))
    return circ


def get_up_to_range_k_paulis(sys_size: int, pauli_range: int, yield_identity: bool = False,
                             pauli_types: List[str] = None) -> List[Pauli]:
    if pauli_types is None:
        pauli_types = ['X', 'Y', 'Z']
    elif any(s not in ['X', 'Y', 'Z'] for s in pauli_types):
        raise Exception('Bad Pauli types')
    paulis = []
    if yield_identity:
        paulis.append(Pauli(''.join(['I'] * sys_size)))
        if pauli_range == 0:
            return paulis
    for site in range(sys_size):
        support_size = min(pauli_range, sys_size - site)
        op_list = [pauli_types] + [['I'] + pauli_types] * (support_size - 1)
        for i, pauli_str in enumerate(itertools.product(*op_list)):
            full_pauli_str = ['I'] * sys_size
            full_pauli_str[site:site + support_size] = pauli_str
            paulis.append(Pauli(''.join(full_pauli_str)))
    return paulis


def get_air_simulator():
    backend = Aer.get_backend('aer_simulator')
    print(f"Using backend: {backend.name()}")
    return backend


def get_density_matrix_from_simulation(circ: QuantumCircuit, qubits_to_reconstruct: List[int],
                                       backend: Optional[Union[AerSimulator, Backend]] =
                                       Aer.get_backend('aer_simulator'), optimization_level: int = 3,
                                       initial_layout: List[int] = None, use_noise_model: bool = True) -> np.array:
    if backend is not None and not isinstance(backend, AerSimulator):
        if use_noise_model:
            backend = Aer.get_backend('aer_simulator').from_backend(backend)
        else:
            backend = Aer.get_backend('aer_simulator')
    circ = circ.copy()
    circ.append(SaveDensityMatrix(len(qubits_to_reconstruct), 'dm'), qubits_to_reconstruct)
    transpiled_circ = transpile(circ, backend, optimization_level=optimization_level, initial_layout=initial_layout)
    return backend.run(transpiled_circ).result().data()['dm']
