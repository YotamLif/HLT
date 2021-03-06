import logging
from collections import defaultdict
from typing import List, Dict, Optional

import numpy as np
from qiskit.quantum_info import Pauli

from utils.qiskit_utils import get_up_to_range_k_paulis


class ConstrainMatrix:
    """

    :param sys_size: number of qubits to reconstruct
    :param range_constraints: the maximum locality of the constraints in the
     constraint matrix (should be bigger than range_hamil_terms, usually taken
     as range_constraints=range_hamil_terms+1)
    :param range_hamil_terms: the maximum locality of the Hamiltonians which
     assumed to be in the GH. These Hamiltonian will define the constraint
     matrix.
    :param hamil_terms: list of Paulis used in the constraint matrix
    :param constraints: list of constraints (Paulis) used in the constraint
     matrix
    :param operator_matrix: a matrix which contains the operators in the
     constraint matrix entries as defined in the main paper. Since the
     Hamiltonians and constraint are Pauli operators with k1 and k2 maximal
     locality, respectively, these entries, which are commutation relations
     between constraints and hamiltonians, are at most (k1+k2-1)-local Paulis
     (usually k2=k1+1).
    :param constraint_matrix: the constraint matrix filled with expectation
     value from experiment or simulation data (after get_constraint_matrix is
     called).
    """
    sys_size: int
    range_hamil_terms: int
    range_constraints: int
    hamil_terms: List[Pauli]
    constraints: List[Pauli]
    operator_matrix: Dict[Pauli, np.array]
    constraint_matrix: Optional[np.array]

    def __init__(self, sys_size: int, range_constraints: int, range_hamil_terms: int) -> None:
        """

        :param sys_size: number of qubits to reconstruct
        :param range_constraints: the maximum locality of the constraints, which
         are taken to be Pauli operators, in the constraint matrix (should be
         bigger than range_hamil_terms, usually taken
         as range_constraints=range_hamil_terms+1).
        :param range_hamil_terms: the maximum locality of the Hamiltonians
         which assumed to be in the GH. These Hamiltonian will define the
         constraint matrix.
        """
        self.sys_size = sys_size
        self.range_constraints = range_constraints
        self.range_hamil_terms = range_hamil_terms
        self.hamil_terms = []
        self.constraints = []
        self.hamil_terms = get_up_to_range_k_paulis(self.sys_size, self.range_hamil_terms)
        self.constraints = get_up_to_range_k_paulis(self.sys_size, self.range_constraints)
        self._init_operator_matrix()
        self.constraint_matrix = None

    def _init_operator_matrix(self):
        self.operator_matrix = defaultdict(self._get_empty_k_matrix)
        logging.debug("Initializing constraint operator matrix:")
        for i, constraint in enumerate(self.constraints):
            for j, hamil_term in enumerate(self.hamil_terms):
                if hamil_term.commutes(constraint):
                    continue
                self._init_term_in_operator_matrix(constraint, hamil_term, i, j)

    def _get_empty_k_matrix(self) -> np.array:
        return np.zeros([len(self.constraints), len(self.hamil_terms)], dtype=np.complex)

    def _init_term_in_operator_matrix(self, constraint: Pauli, hamil_term: Pauli, i: int, j: int) -> None:
        first_part_of_commutation_relations = constraint.dot(hamil_term)
        first_phase = (-1j) ** first_part_of_commutation_relations.phase
        first_part_of_commutation_relations.phase = 0
        second_part_of_commutation_relations = hamil_term.dot(constraint)
        second_phase = (-1) * ((-1j) ** second_part_of_commutation_relations.phase)
        second_part_of_commutation_relations.phase = 0
        self.operator_matrix[first_part_of_commutation_relations][i][j] += 1j * first_phase
        self.operator_matrix[second_part_of_commutation_relations][i][j] += 1j * second_phase

    def get_paulis_in_constraint_matrix(self) -> List[Pauli]:
        """

        :return: all the Paulis in the constraint matrix entries
        """
        return list(self.operator_matrix.keys())

    def get_constraint_matrix(self, expectations: Dict[Pauli, complex]) -> np.array:
        """
        Fill the constraint matrix with expectation values from simulation or
        experiment

        :param expectations: a map from Pauli operator to its expectation value
         in the experiment or simulation
        :return: a constraint matrix, as defined in the main text, with the
         given expectation values.
        """
        constraint_matrix = self._get_empty_k_matrix()
        logging.debug("Getting constraint matrix")
        for pauli, coefficients in self.operator_matrix.items():
            # coefficients is a matrix of coefficients, making sure expectation values are real
            constraint_matrix += np.real(coefficients * expectations[pauli])
        self.constraint_matrix = constraint_matrix
        return self.constraint_matrix

    def get_constraint_matrix_hamiltonians(self, num_hamiltonians: int) -> List[np.array]:
        """

        :param num_hamiltonians:
        :return: the Hamiltonians defined by the num_hamiltonians lowest
         singular vectors of the constraint matrix (see main text for details)
        """
        if self.constraint_matrix is None:
            raise Exception("Constraint matrix is None but must be calculated before getting terms")
        u, d, v = np.linalg.svd(self.constraint_matrix)
        logging.debug(f"Constraint matrix have {len(d)} singular values.")
        terms = []
        for i in range(len(d)):
            if i < num_hamiltonians:
                terms.append(self._get_constraint_matrix_hamiltonian_from_singular_vector(v[-(i + 1), :]))
        return terms

    def _get_constraint_matrix_hamiltonian_from_singular_vector(self, v: np.array):
        term = np.zeros_like(self.hamil_terms[0])
        for coefficient, h in zip(v, self.hamil_terms):
            term += coefficient * h.to_matrix()
        return term
