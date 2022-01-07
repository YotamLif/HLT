from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, ItemsView, Callable, Tuple

import numpy as np
from qiskit.quantum_info import Pauli, DensityMatrix

from utils.np_utils import get_gibbs_hamiltonian, get_fidelity
from utils.hamil_utils import get_density_matrix_from_gibbs_hamiltonian
from utils.qiskit_utils import get_up_to_range_k_paulis


@dataclass
class PauliDecomposition:
    sys_size: int
    operator: np.array
    decomposition: Dict[Pauli, float]

    def __init__(self, sys_size: int, operator: np.array, decomposition: Dict[Pauli, float]) -> None:
        self.sys_size = sys_size
        self.operator = operator
        self.decomposition = decomposition

    @classmethod
    def pauli_decomposition(cls, sys_size: int, operator: np.array, paulis: Iterable[Pauli]):
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
        return PauliDecomposition.pauli_decomposition(sys_size, operator,
                                                      get_up_to_range_k_paulis(sys_size, k, yield_identity=True))

    @classmethod
    def pauli_exactly_range_k_decomposition(cls, sys_size: int, operator: np.array, k: int = 2):
        if k == 0:
            return PauliDecomposition.pauli_decomposition(sys_size, operator,
                                                          get_up_to_range_k_paulis(sys_size, k, yield_identity=True))
        paulis = list(set(get_up_to_range_k_paulis(sys_size, k, yield_identity=True))
                      .difference(get_up_to_range_k_paulis(sys_size, k - 1, yield_identity=True)))
        return PauliDecomposition.pauli_decomposition(sys_size, operator, paulis)

    def get_not_normalized_str(self) -> str:
        s = ''
        for pauli, value in sorted(self.decomposition.items(), key=lambda item: item[1], reverse=True):
            s += f'{str(pauli)}: {value:.5f}\n'
        return s[:-1]

    def get_not_normalized_str_top_coefficients(self, n: int) -> str:
        s = ''
        top_l = list(sorted(self.decomposition.items(), key=lambda item: np.abs(item[1]), reverse=True)[0:n])
        for pauli, value in sorted(top_l, key=lambda item: item[1], reverse=True):
            s += f'{str(pauli)}: {value:.5f}\n'
        return s[:-1]

    def get_normalized_coefficients_squared_str(self) -> str:
        s = ''
        decomposition_sum = np.sum(np.abs(self.operator))
        for pauli, value in sorted(self.decomposition.items(), key=lambda item: abs(item[1]) ** 2 / decomposition_sum,
                                   reverse=True):
            s += f'{str(pauli)}: {value / decomposition_sum:.5f}\n'
        return s[:-1]

    def get_classical_quantum_parts(self) -> Tuple[float, float, float, float, float]:
        operator_norm = (np.linalg.norm(self.operator) ** 2)
        classical1 = self._filter_decomposition_norm(lambda item: str(item[0]).count('Z') == 1
                                                                  and str(item[0]).count('X') == 0
                                                                  and str(item[0]).count('Y') == 0) / operator_norm
        classical2 = self._filter_decomposition_norm(lambda item: str(item[0]).count('Z') == 2
                                                                  and str(item[0]).count('X') == 0
                                                                  and str(item[0]).count('Y') == 0
                                                                  and self._locality_of_pauli(
            item[0]) == 2) / operator_norm
        quantum1 = self._filter_decomposition_norm(lambda item: str(item[0]).count('Z') == 0
                                                                and str(item[0]).count('X')
                                                                + str(item[0]).count('Y') == 1) / operator_norm
        quantum2 = self._filter_decomposition_norm(lambda item: str(item[0]).count('Z')
                                                                + str(item[0]).count('X')
                                                                + str(item[0]).count('Y') == 2
                                                                and str(item[0]).count('X')
                                                                + str(item[0]).count('Y') >= 1
                                                                and self._locality_of_pauli(
            item[0]) == 2) / operator_norm
        higher = self._filter_decomposition_norm(lambda item: self._locality_of_pauli(item[0]) > 2) / operator_norm
        return classical1, classical2, quantum1, quantum2, higher

    @staticmethod
    def _locality_of_pauli(pauli: Pauli):
        pauli_str = str(pauli)
        start = min([PauliDecomposition._index_of(pauli_str, 'X'),
                     PauliDecomposition._index_of(pauli_str, 'Y'),
                     PauliDecomposition._index_of(pauli_str, 'Z')])
        pauli_str = pauli_str[::-1]
        end = len(pauli_str) - 1 - min([PauliDecomposition._index_of(pauli_str, 'X'),
                                        PauliDecomposition._index_of(pauli_str, 'Y'),
                                        PauliDecomposition._index_of(pauli_str, 'Z')])
        if start == sys.maxsize or end == sys.maxsize:
            return 0
        return end - start + 1

    @staticmethod
    def _index_of(s: str, sub_s: str):
        try:
            return s.index(sub_s)
        except ValueError:
            return sys.maxsize

    def _filter_decomposition_norm(self, condition: Callable) -> np.array:
        filtered_decomposition = filter(condition, self.decomposition.items())
        filtered_decomposition = PauliDecomposition(self.sys_size, self.operator,
                                                    {k: v for k, v in filtered_decomposition})
        return filtered_decomposition.norm()

    def density_matrix_composition(self):
        return get_density_matrix_from_gibbs_hamiltonian(self.gibbs_hamiltonian_composition())

    def gibbs_hamiltonian_composition(self):
        gibbs_hamiltonian = np.zeros((2 ** self.sys_size, 2 ** self.sys_size), dtype=np.cdouble)
        for pauli, coef in self.decomposition.items():
            # noinspection PyTypeChecker
            gibbs_hamiltonian = gibbs_hamiltonian + pauli.to_matrix() * coef / np.sqrt(pauli.dim[0])
        return gibbs_hamiltonian

    def get_fidelity_from_decomposition(self, density_matrix: np.array) -> float:
        composition_density_matrix = self.density_matrix_composition()
        f = get_fidelity(density_matrix, composition_density_matrix)
        return f

    def items(self) -> ItemsView:
        return self.decomposition.items()

    def get_decomposition_part(self) -> float:
        decomposition_part = self.norm()**2 / (np.linalg.norm(self.operator) ** 2)
        return decomposition_part

    def diff(self, deco2: PauliDecomposition):
        s = 0
        for p in set(self.decomposition.keys()).union(set(deco2.decomposition.keys())):
            s += (self.decomposition.get(p, 0) - deco2.decomposition.get(p, 0)) ** 2
        return np.sqrt(s)

    def norm(self):
        s = 0
        for p in self.decomposition.keys():
            s += self.decomposition.get(p, 0) ** 2
        return np.sqrt(s)


def analyze_decomposition(density_matrix: DensityMatrix, decomposition: PauliDecomposition, print_results: bool = False) \
        -> (float, float):
    print("Decomposition is:") if print_results else None
    # print(decomposition.get_not_normalized_str()) if print_results else None
    print(decomposition.get_not_normalized_str_top_coefficients(10)) if print_results else None
    # print("Decomposition normalized is:") if print_results else None
    # print(decomposition.get_normalized_coefficients_squared_str()) if print_results else None
    c1, c2, q1, q2, higher = decomposition.get_classical_quantum_parts()
    print(f"c1: {c1}") if print_results else None
    print(f"c2: {c2}") if print_results else None
    print(f"q1: {q1}") if print_results else None
    print(f"q2: {q2}") if print_results else None
    print(f"higher locality: {higher}") if print_results else None
    decomposition_part = decomposition.get_decomposition_part()
    print(f"Decomposition part is: {decomposition_part}") if print_results else None
    f = decomposition.get_fidelity_from_decomposition(density_matrix)
    print(f"Fidelity is: {f}") if print_results else None
    return decomposition_part, f


def analyze_density_matrix(density_matrix: np.array, sys_size: int, pauli_range: int = 5, remove_identity: bool = False,
                           print_results: bool = False) -> (float, float):
    gibbs_hamiltonian = get_gibbs_hamiltonian(density_matrix)
    if remove_identity:
        gibbs_hamiltonian -= np.identity(2 ** sys_size) * np.trace(gibbs_hamiltonian) / (2 ** sys_size)
    decomposition = PauliDecomposition.pauli_up_to_range_k_decomposition(sys_size, gibbs_hamiltonian, k=pauli_range)
    return analyze_decomposition(density_matrix, decomposition, print_results)
