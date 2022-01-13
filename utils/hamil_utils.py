import abc
import random
from typing import List, Optional, Union

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.opflow import PauliOp
from qiskit.quantum_info import Pauli, DensityMatrix, Statevector
from scipy.linalg import expm

from utils.np_utils import normalized_matrix
from utils.qiskit_utils import get_density_matrix_circuit, get_up_to_range_k_paulis


class Hamiltonian(metaclass=abc.ABCMeta):
    """
    Abstract class for building operators which represent Hamiltonians

    :param sys_size: number of qubits the Hamiltonian is acting upon
    :param matrix: the matrix representation of the Hamiltonian.
    """
    sys_size: int
    matrix: np.array = None

    @abc.abstractmethod
    def to_matrix(self) -> np.array:
        """

        :return: the matrix representation of the Hamiltonian
        """
        raise NotImplementedError()

    def get_ground_state(self) -> np.array:
        """

        :return: the ground state of the Hamiltonian (as a vector)
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.to_matrix())
        return eigenvectors[:, np.argmin(eigenvalues)]

    def get_ground_state_density_matrix(self) -> DensityMatrix:
        """

        :return: returns the ground state of the Hamiltonian as a state vector
        """
        return DensityMatrix(Statevector(np.asarray(self.get_ground_state())))


class RandomPauliHamiltonian(Hamiltonian):
    """
    A Hamiltonian which is made from Pauli operators
    """
    paulis: List[PauliOp]

    def __init__(self, paulis: Union[List[Pauli], List[PauliOp]], rand_bot: float = -1, rand_top: float = 1) -> None:
        """

        :param paulis: the Paulis which the Hamiltonian is made of, each will
         have a random coefficient between rand_bot and rand_top.
        :param rand_bot: minimal coefficient value for Hamiltonian terms
        :param rand_top: maximal coefficient value for Hamiltonian terms
        """
        super().__init__()
        if len(paulis) == 0:
            raise ValueError("Hamiltonian must contain at least one Pauli!")
        if isinstance(paulis[0], Pauli):
            paulis = [PauliOp(p, random.uniform(rand_bot, rand_top)) for p in paulis]
        self.sys_size = paulis[0].num_qubits
        self.paulis = paulis

    @classmethod
    def init_from_str(cls, s: str):
        """

        :param s:  a string which contains the Hamiltonian terms, each in a new
         line. Each line starts with a pauli string followed by : and the
         coefficient of the term in the Hamiltonian
        :return: :class:`.PauliHamiltonian` defined by the string
        """
        paulis = []
        for line in str.splitlines(s):
            split = line.split(":")
            pauli = PauliOp(Pauli(split[0].strip()), float(split[1].strip()))
            paulis.append(pauli)
        return cls(paulis)

    def __str__(self) -> str:
        s = ''
        for p in self.paulis:
            s += f'{p.primitive.to_label()}:{p.coeff}\n'
        return s

    def to_matrix(self) -> np.array:
        """

        :return: the matrix representation of the Hamiltonian
        """
        if self.matrix is not None:
            return self.matrix
        H = self.paulis[0].to_spmatrix() * self.paulis[0].coeff
        for i in range(1, len(self.paulis)):
            H = H + self.paulis[i].to_spmatrix() * self.paulis[i].coeff
        H = H.todense()
        self.matrix = H
        return self.matrix

    @staticmethod
    def _pauli(p: Union[Pauli, List[str]], coefficient: float) -> PauliOp:
        if isinstance(p, Pauli):
            return PauliOp(p, coefficient)
        return PauliOp(Pauli(''.join(p)), coefficient)


class RandomKLocalHamiltonian(RandomPauliHamiltonian):
    """
    :class:`.PauliHamiltonian` which contains all the k-local Pauli operators
    (of the same type)
    """
    def __init__(self, sys_size: int, k: int, pauli_types: Optional[List[str]] = None) -> None:
        """

        :param sys_size: number of qubits in the system
        :param k: the maximal locality of the Paulis
        :param pauli_types: the types of Pauli allowed, if None then all
         Paulis are allowed (X,Y,Z)
        """
        paulis = get_up_to_range_k_paulis(sys_size, k, pauli_types=pauli_types)
        paulis = [self._pauli(p, random.uniform(-2, 2)) for p in paulis]
        super().__init__(paulis)


class TransverseIsingHamiltonian(RandomPauliHamiltonian):
    """
    Transverse Ising Hamiltonian, as defined in the main text
    """
    B: float
    J: float

    def __init__(self, sys_size: int, B: float, J: float = 1) -> None:
        """

        :param sys_size: number of qubits in the system
        :param B: the coefficient of (all) the :math:`Z_i` terms in the
         Hamiltonian
        :param J: the coefficient of (all) the :math:`X_i X_j` terms in the
         Hamiltonian
        """
        self.sys_size = sys_size
        # Z coefficient
        self.B = B
        # XX coefficient
        self.J = J
        super().__init__(self._get_xx_terms() + self._get_z_terms())

    def _get_xx_terms(self) -> List[PauliOp]:
        paulis = []
        for i in range(self.sys_size - 1):
            p_str = ['I'] * self.sys_size
            p_str[i] = 'X'
            p_str[i + 1] = 'X'
            paulis.append(self._pauli(p_str, self.J))
        return paulis

    def _get_z_terms(self) -> List[PauliOp]:
        paulis = []
        for i in range(self.sys_size):
            p_str = ['I'] * self.sys_size
            p_str[i] = 'Z'
            paulis.append(self._pauli(p_str, self.B))
        return paulis


def get_ghz_circ(sys_size: int) -> QuantumCircuit:
    """

    :param sys_size: number of qubits
    :return: a circuit which produces GHZ state (a.k.a cat state) on sys_size
     qubits
    """
    circ = QuantumCircuit(sys_size)
    circ.h(0)
    for i in range(1, sys_size):
        circ.cx(0, i)
    return circ


def get_random_k_local_gibbs_circ(sys_size: int, k: int) -> QuantumCircuit:
    """

    :param sys_size: the number of qubits in the system.
    :param k: the maximal locality of terms in the Gibbs Hamiltonian
    :return: a circuit which produces, on a Qiskit simulation, a k-local gibbs
     state with random coefficients for each term between -1 to 1.
    """
    gibbs_hamiltonian = RandomKLocalHamiltonian(sys_size, k).to_matrix()
    density_matrix = get_density_matrix_from_gibbs_hamiltonian(gibbs_hamiltonian)
    return get_circ_from_gibbs_hamiltonian(sys_size, density_matrix)


def get_density_matrix_from_gibbs_hamiltonian(gibbs_hamiltonian: Union[np.array, torch.Tensor]) -> \
        Union[np.array, torch.Tensor]:
    """

    :param gibbs_hamiltonian: the Gibbs Hamiltonian which defines the Gibbs
     state.
    :return: the density matrix of a Gibbs state defined by the Gibbs
     Hamiltonian (beta=1)
    """
    if isinstance(gibbs_hamiltonian, torch.Tensor):
        gibbs_hamiltonian_exp = torch.matrix_exp(-gibbs_hamiltonian)
    else:
        gibbs_hamiltonian_exp = expm(-gibbs_hamiltonian)
    return normalized_matrix(gibbs_hamiltonian_exp)


def get_circ_from_gibbs_hamiltonian(sys_size: int, gibbs_hamiltonian: Union[np.array, torch.Tensor]) -> \
        Union[np.array, torch.Tensor]:
    """

    :param sys_size: number of qubits in the system
    :param gibbs_hamiltonian: the Gibbs Hamiltonian to produce
    :return: a circuit which produces a Gibbs state defined by gibbs_hamiltonian
     (beta=1) on a Qiskit simulation.
    """
    density_matrix = get_density_matrix_from_gibbs_hamiltonian(gibbs_hamiltonian)
    return get_density_matrix_circuit(sys_size, density_matrix)
