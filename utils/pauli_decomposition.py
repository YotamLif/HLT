from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, ItemsView

import numpy as np
from qiskit.quantum_info import Pauli, DensityMatrix

from utils.hamil_utils import get_density_matrix_from_gibbs_hamiltonian
from utils.np_utils import get_gibbs_hamiltonian, get_fidelity
from utils.qiskit_utils import get_up_to_range_k_paulis


@dataclass
class PauliDecomposition:
    """
    A pauli decomposition of a matrix, defined by the inner product
    :math:`<A,B>=(1/2**N) * Trace(A*B)` and from the fact that Pauli matrices are a basis
    of the hermitian matrices.
    """
    sys_size: int
    operator: np.array
    decomposition: Dict[Pauli, float]

    def __init__(self, sys_size: int, operator: np.array, decomposition: Dict[Pauli, float]) -> None:
        """

        :param sys_size: number of qubits in the system
        :param operator: the decomposed matrix
        :param decomposition: a map from Pauli to coefficient which defines
         the decomposition, i.e,
         (1/2**N) * Trace(operator*Pauli)=decomposition[Pauli]
        """
        self.sys_size = sys_size
        self.operator = operator
        self.decomposition = decomposition

    @classmethod
    def pauli_decomposition(cls, sys_size: int, operator: np.array, paulis: Iterable[Pauli]):
        """

        :param sys_size: number of qubits in the system
        :param operator: the matrix to decompose
        :param paulis: which Paulis to decompose with
        :return: the decomposition of the operator defined by the given Paulis
        """
        decomposition = {}
        for pauli in paulis:
            inner_product = np.trace(operator @ pauli.to_matrix())
            if np.imag(inner_product) > 1e-8:
                warnings.warn(f"Inner product should be real, inner product is: {inner_product}")
            inner_product = np.real(inner_product)
            inner_product /= np.sqrt(pauli.dim[0])
            decomposition[pauli] = inner_product.item()
        return cls(sys_size, operator, decomposition)

    @classmethod
    def pauli_up_to_range_k_decomposition(cls, sys_size: int, operator: np.array, k: int = 2):
        """

        :param sys_size: number of qubits in the system
        :param operator: the matrix to decompose
        :param k: the maximal locality of Paulis to decompose with
        :return: up k-local Pauli decomposition of the operator.
        """
        return PauliDecomposition.pauli_decomposition(sys_size, operator,
                                                      get_up_to_range_k_paulis(sys_size, k, yield_identity=True))

    @classmethod
    def pauli_exactly_range_k_decomposition(cls, sys_size: int, operator: np.array, k: int = 2):
        """

        :param sys_size: number of qubits in the system
        :param operator: the matrix to decompose
        :param k: the locality of Paulis to decompose with
        :return: k-local Pauli decomposition of the operator.
        """
        if k == 0:
            return PauliDecomposition.pauli_decomposition(sys_size, operator,
                                                          get_up_to_range_k_paulis(sys_size, k, yield_identity=True))
        paulis = list(set(get_up_to_range_k_paulis(sys_size, k, yield_identity=True))
                      .difference(get_up_to_range_k_paulis(sys_size, k - 1, yield_identity=True)))
        return PauliDecomposition.pauli_decomposition(sys_size, operator, paulis)

    def get_not_normalized_str(self) -> str:
        """

        :return: the decomposition as a string. Each line contains a Pauli and
         the matching coefficient (i.e (1/2**N) *  Trace(operator, Pauli))
         sorted reversely by the coefficient.
        """
        s = ''
        for pauli, value in sorted(self.decomposition.items(), key=lambda item: item[1], reverse=True):
            s += f'{str(pauli)}: {value:.5f}\n'
        return s[:-1]

    def get_not_normalized_str_top_coefficients(self, n: int) -> str:
        """

        :param n: the number of coefficients to return
        :return: same as get_not_normalized_str, but only the n largest
         coefficients (considering absolute vaule)
        """
        s = ''
        top_l = list(sorted(self.decomposition.items(), key=lambda item: np.abs(item[1]), reverse=True)[0:n])
        for pauli, value in sorted(top_l, key=lambda item: item[1], reverse=True):
            s += f'{str(pauli)}: {value:.5f}\n'
        return s[:-1]

    def get_normalized_coefficients_squared_str(self) -> str:
        """

        :return: the normalized (normalization defined by Parseval theorem)
         decomposition as a string. Each line contains a Pauli and
         the matching coefficient (i.e [(1/2**N) *  Trace(operator, Pauli)]**2)
         sorted reversely by the coefficient.
        """
        s = ''
        decomposition_sum = np.sum(np.abs(self.operator))
        for pauli, value in sorted(self.decomposition.items(), key=lambda item: abs(item[1]) ** 2 / decomposition_sum,
                                   reverse=True):
            s += f'{str(pauli)}: {value / decomposition_sum:.5f}\n'
        return s[:-1]

    def gibbs_hamiltonian_composition(self):
        """

        :return: the Gibbs Hamiltonian defined by the decomposition (i.e
         multiplying each Pauli by its coefficient)
        """
        gibbs_hamiltonian = np.zeros((2 ** self.sys_size, 2 ** self.sys_size), dtype=np.cdouble)
        for pauli, coef in self.decomposition.items():
            # noinspection PyTypeChecker
            gibbs_hamiltonian = gibbs_hamiltonian + pauli.to_matrix() * coef / np.sqrt(pauli.dim[0])
        return gibbs_hamiltonian

    def density_matrix_composition(self):
        """

        :return: the density matrix of the Gibbs state of the Gibbs Hamiltonian
         defined by the decomposition (see gibbs_hamiltonian_composition)
        """
        return get_density_matrix_from_gibbs_hamiltonian(self.gibbs_hamiltonian_composition())

    def get_fidelity_from_decomposition(self, density_matrix: np.array) -> float:
        """

        :param density_matrix: the density matrix to compare with
        :return: the fidelity between density_matrix and
         density_matrix_composition
        """
        composition_density_matrix = self.density_matrix_composition()
        f = get_fidelity(density_matrix, composition_density_matrix)
        return f

    def items(self) -> ItemsView:
        """

        :return: the items of the decomposition, the key is the Pauli, and the
         value is the coefficient
        """
        return self.decomposition.items()

    def get_decomposition_part(self) -> float:
        """

        :return: the sum of the decomposition coefficients squared divided by
         the operator norm squared. This normalization is defined by the
         Parseval theorem.
        """
        decomposition_part = self.norm() ** 2 / (np.linalg.norm(self.operator) ** 2)
        return decomposition_part

    def norm(self):
        """

        :return: the square rott of the sum of coefficient squared
        """
        s = 0
        for p in self.decomposition.keys():
            s += self.decomposition.get(p, 0) ** 2
        return np.sqrt(s)


def _analyze_decomposition(density_matrix: DensityMatrix, decomposition: PauliDecomposition) -> (float, float):
    decomposition_part = decomposition.get_decomposition_part()
    f = decomposition.get_fidelity_from_decomposition(density_matrix)
    return decomposition_part, f


def analyze_density_matrix(density_matrix: np.array, sys_size: int, pauli_range: int) -> (float, float):
    """
    Decompose density_matrix using Paulis with maximal pauli_range locality
    and analyze this decomposition.

    :param density_matrix: the density matrix to analyze, which fits the
     Gibbs state for the decomposed Gibbs Hamiltonian given in decomposition.
    :param sys_size: number of qubits in the system
    :param pauli_range: the maximal locality of Hamiltonian to decompose
     density_matrix's Gibbs Hamiltonian with.
    :return: A tuple which includes the decomposition part
     as defined in :attr:`.PauliDecomposition.decomposition_part` and the
     fidelity between density_matrix and the density matrix produced by taking
     Gibbs state of this decomposition's Gibbs Hamiltonian.
    """
    gibbs_hamiltonian = get_gibbs_hamiltonian(density_matrix)
    decomposition = PauliDecomposition.pauli_up_to_range_k_decomposition(sys_size, gibbs_hamiltonian, k=pauli_range)
    return _analyze_decomposition(density_matrix, decomposition)
