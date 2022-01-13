from typing import Union, Optional, List

from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.providers.aer import AerSimulator

from tomography.ansatz import AnsatzResults, get_ansatz
from tomography.cyclic_measurer import CyclicMeasurer
from utils.hamil_utils import TransverseIsingHamiltonian, get_ghz_circ, \
    get_random_k_local_gibbs_circ, get_circ_from_gibbs_hamiltonian
from utils.np_utils import get_fidelity
from utils.pauli_decomposition import analyze_density_matrix
from utils.qiskit_utils import get_density_matrix_from_simulation, get_air_simulator


def analyze_circ(sys_size: int,
                 circ: QuantumCircuit,
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
    """

    :param sys_size: number of qubits in the circuits
    :param circ: circuits which produces the wanted density matrix
    :param qubits_to_reconstruct: which qubits to reconstruct the density matrix
     for with HLT
    :param backend: which backend to measure expectation values with.
    :param range_constraints: the maximum locality of the constraints, which
     are taken to be Pauli operators, in the constraint matrix (should be
     bigger than range_hamil_terms, usually taken
     as range_constraints=range_hamil_terms+1).
    :param range_hamiltonians: the maximum locality of the Hamiltonians
     which assumed to be in the GH. These Hamiltonian will define the
     constraint matrix.
    :param num_parameters: number of parameters to used in HLT ansatz (denoted
     as `l` in the main text graphs).
    :param total_number_of_shots: total number of shots to use to measure
     the expectation values with.
    :param optimization_level: How much optimization to perform on the circuits,
     as described in qiskit.compiler.transpile
    :param initial_layout: Initial position of virtual qubits on physical qubits,
     as described in qiskit.compiler.transpile
    :param is_torch: whether to use PyTorch optimization or SciPy optimization,
     see main text for more details
    :return: an AnsatzResults with the following data:
     the cyclic measurer used for HLT, the ansatz produced by HLT, the final
     loss of HLT optimization, the HLT reconstructed density matrix, the
     ground truth density matrix calculated with Qiskit simulation and the
     fidelity between the reconstructed density matrix and the ground truth
     density matrix (can also be saved in Pickle file).
    """
    print(f"Sys_size: {sys_size}, range constraints: {range_constraints}, range hamil terms: {range_hamiltonians},"
          f" number of parameters in ansatz: {num_parameters}")
    cyclic_measurer = CyclicMeasurer(qubits_to_reconstruct, circ, backend)
    cyclic_measurer_results = cyclic_measurer.get_cyclic_measurements_results(
        CyclicMeasurer.get_cycle_size(range_hamiltonians, range_constraints),
        total_number_of_shots, optimization_level, initial_layout)
    ground_truth_density_matrix = get_density_matrix_from_simulation(circ, qubits_to_reconstruct, backend,
                                                                     optimization_level=optimization_level,
                                                                     initial_layout=initial_layout)
    deco_part, deco_f = analyze_density_matrix(ground_truth_density_matrix,
                                               len(qubits_to_reconstruct),
                                               pauli_range=range_hamiltonians)
    print(f"Decomposition part for ground truth density matrix of qubits {qubits_to_reconstruct} "
          f"with {range_hamiltonians}-local Paulis is: {deco_part}\n with fidelity: {deco_f}")
    ansatz = get_ansatz(qubits_to_reconstruct, is_torch, num_parameters, range_constraints, range_hamiltonians,
                        cyclic_measurer_results)
    forward = ansatz.forward()
    f = get_fidelity(forward, ground_truth_density_matrix)
    print(f"Fidelity between ansatz and ground truth density_matrix: {f}")
    loss = ansatz.loss()
    result = AnsatzResults(cyclic_measurer, ansatz, loss, ground_truth_density_matrix, f)
    return result


def main_ghz(sys_size: int) -> None:
    """
    An example of HLT reconstruction of density matrix of N out of N+1 qubits
    in the GHZ state on simulation.

    :param sys_size: number of qubits in the system
    :return:
    """
    print("GHZ state")
    circ = get_ghz_circ(sys_size)
    analyze_circ(sys_size, circ, list(range(sys_size - 1)), get_air_simulator(), 3, 2, 30, 100000)


def main_random_k_local_gibbs(sys_size: int, k: int) -> None:
    """
    An example of HLT reconstruction of Gibbs state of a random Gibbs
    Hamiltonian with coefficients in [-1,1] and only up to k-local Pauli,
    assuming a 1D open chain topology.

    :param sys_size: number of qubits in the system
    :param k: the locality of the random Gibbs Hamiltonian
    :return:
    """
    print("Gibbs state")
    circ = get_random_k_local_gibbs_circ(sys_size, k)
    analyze_circ(sys_size, circ, list(range(sys_size)), get_air_simulator(), 3, 2, 30, 100000)


def main_transverse_ising(sys_size: int, J: float = 1, B: float = 1.5):
    """
    An example of HLT reconstruction of Gibbs state of a Transverse Ising Gibbs
    Hamiltonian with J and B (as defined in the main text),
    assuming a 1D open chain topology.

    :param sys_size: number of qubits in the system
    :param B: the coefficient of (all) the :math:`Z_i` terms in the
     Transverse Ising Hamiltonian
    :param J: the coefficient of (all) the :math:`X_i X_j` terms in the
     Transverse Ising Hamiltonian
    :return:
    """
    print("Transverse Ising model")
    gibbs_hamiltonian = TransverseIsingHamiltonian(sys_size, B=B, J=J)
    circ = get_circ_from_gibbs_hamiltonian(sys_size, gibbs_hamiltonian.to_matrix())
    analyze_circ(sys_size, circ, list(range(sys_size)), get_air_simulator(), 3, 2, 30, 100000)


if __name__ == '__main__':
    """
    Choose one of the simulations above, enter the needed parameters, and run
    to get printed results example of HLT use cases.
    """
    main_random_k_local_gibbs(3,2)
