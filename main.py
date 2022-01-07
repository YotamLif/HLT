import logging
from typing import Union, Callable, Optional, List

from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.providers.aer import AerSimulator

from tomography.ansatz import AnsatzResults, get_ansatz
from tomography.cyclic_measure import CyclicMeasurer
from utils.hamil_utils import TransverseIsingHamiltonian, get_ghz_circ, \
    get_random_k_local_gibbs_circ, get_circ_from_gibbs_hamiltonian
from utils.np_utils import get_fidelity
from utils.pauli_decomposition import analyze_density_matrix
from utils.qiskit_utils import get_density_matrix_from_simulation, get_air_simulator


def analyze_circ(sys_size: int,
                 circ: Union[QuantumCircuit, Callable[[], QuantumCircuit]],
                 qubits_to_reconstruct: List[int],
                 backend: Union[AerSimulator, Backend],
                 range_constraints: int,
                 range_hamiltonians: int,
                 num_parameters: int,
                 total_number_of_shots: int,
                 optimization_level: int = 3,
                 initial_layout: Optional[List[int]] = None,
                 is_torch: bool = False,
                 ) -> AnsatzResults:
    print(f"Sys_size: {sys_size}, range constraints: {range_constraints}, range hamil terms: {range_hamiltonians},"
          f" number of parameters in ansatz: {num_parameters}")
    cyclic_measurer = CyclicMeasurer(qubits_to_reconstruct, circ, backend)
    cyclic_measurer_results = cyclic_measurer.get_cyclic_measurements_results(
        CyclicMeasurer.get_cycle_size(range_hamiltonians, range_constraints),
        total_number_of_shots, optimization_level, initial_layout)
    ground_truth_density_matrix = get_density_matrix_from_simulation(circ, qubits_to_reconstruct, backend,
                                                                     optimization_level=optimization_level,
                                                                     initial_layout=initial_layout)
    deco_part, deco_f = analyze_density_matrix(ground_truth_density_matrix, len(qubits_to_reconstruct),
                                               pauli_range=range_hamiltonians,
                                               print_results=False, remove_identity=True)
    print(f"Decomposition for ground truth density matrix of qubits {qubits_to_reconstruct} "
          f"with {range_hamiltonians}-local Paulis is: {deco_part}\n with fidelity: {deco_f}")
    ansatz = get_ansatz(qubits_to_reconstruct, is_torch, num_parameters, range_constraints, range_hamiltonians,
                        cyclic_measurer_results)
    forward = ansatz.forward()
    f = get_fidelity(forward, ground_truth_density_matrix)
    print(f"Fidelity between ansatz and ground truth density_matrix: {f}")
    loss = ansatz.loss()
    result = AnsatzResults(cyclic_measurer, ansatz, loss, ground_truth_density_matrix, f)
    return result


def main_ghz(sys_size: int):
    print("GHZ state")
    circ = get_ghz_circ(sys_size)
    analyze_circ(sys_size, circ, list(range(sys_size-1)), get_air_simulator(), 3, 2, 30, 100000)


def main_random_k_local_gibbs(sys_size: int, k: int):
    print("Gibbs state")
    circ = get_random_k_local_gibbs_circ(sys_size, k)
    analyze_circ(sys_size, circ, list(range(sys_size)), get_air_simulator(), 3, 2, 30, 100000)


def main_transverse_ising(sys_size: int):
    print("Transverse Ising model")
    gibbs_hamiltonian = TransverseIsingHamiltonian(sys_size, B=1.5)
    circ = get_circ_from_gibbs_hamiltonian(sys_size, gibbs_hamiltonian.to_matrix())
    analyze_circ(sys_size, circ, list(range(sys_size)), get_air_simulator(), 3, 2, 30, 100000)


if __name__ == '__main__':
    main_transverse_ising(6)
