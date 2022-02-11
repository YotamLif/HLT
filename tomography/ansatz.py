import abc
import logging
import time
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple, Counter

import numpy as np
import qiskit.quantum_info as qi
import torch
import torch.nn as nn
from qiskit import Aer
from qiskit.quantum_info import Pauli
from scipy.optimize import least_squares

from tomography.constraint_matrix import ConstrainMatrix
from tomography.cyclic_measurer import CyclicMeasurer
from tomography.cyclic_measurer import PauliBasis
from utils.hamil_utils import get_density_matrix_from_gibbs_hamiltonian


@dataclass
class Parameter:
    """
    A class used as an ansatz parameter

    :param operator: the operator multiplied by the parameter coefficient
    :param coefficient: the coefficient of the parameter
    :param name: parameter name
    """
    operator: np.array
    coefficient: float
    name: str = ''

    def get_parameter(self) -> np.array:
        """

        :return: the full operator parameter, coefficient * operator
        """
        return self.operator * self.coefficient

    def __str__(self) -> str:
        return f"{self.name}: {self.coefficient}\n"

    def __repr__(self) -> str:
        return self.__str__()


class Ansatz(abc.ABC):
    sys_size: int
    parameters: List[Parameter]
    data: Dict[Union[PauliBasis, Tuple[str]], Counter]
    range_hamil_terms: int
    range_constraints: int
    constraint_matrix: Optional[ConstrainMatrix]
    num_parameters: int

    def __init__(self, sys_size: int, range_hamil_terms: int, range_constraints: int,
                 data: Dict[Union[PauliBasis, Tuple[str]], Counter], num_parameters: int) -> None:
        """
        The abstract class for all the ansatzes, including the optimization process

        :param sys_size: number of qubits
        :param range_hamil_terms: range of Hamiltonian terms in the constraint matrix
         as defined in the main paper
        :param range_constraints: range of constraints in the constraint matrix as
         defined in the main paper
        :param data: dictionary from Pauli basis to the measurement results of that basis
        :param num_parameters: number of parameters to be used in the ansatz (denoted
         as 'l' in the main paper graphs)
        """
        self.sys_size = sys_size
        self.data = data
        self.constraint_matrix = None
        self.num_parameters = num_parameters
        logging.debug(f"Number of parameters in ansatz is: {num_parameters}")
        self.range_hamil_terms = range_hamil_terms
        self.range_constraints = range_constraints
        self.parameters = []

    @abc.abstractmethod
    def get_gibbs_hamiltonian_from_ansatz(self):
        """
        :return: the Gibbs Hamiltonian which the ansatz defines, as described
         in the main paper.
        """
        ...

    def forward(self) -> np.array:
        """
        :return: the density matrix defined by the Gibbs Hamiltonian of the ansatz
         (see also get_gibbs_hamiltonian_from_ansatz)
        """
        return get_density_matrix_from_gibbs_hamiltonian(self.get_gibbs_hamiltonian_from_ansatz())

    def get_expectation_values_from_data(self, paulis: List[Pauli]) -> Dict[Pauli, complex]:
        """
        :param paulis: Paulis which we need expectation values for constructing
         the constraint matrix
        :return: the expectation values for these paulis in a dictionary
         ordered by the Pauli bases
        """
        counts = defaultdict(complex)
        logging.info("Calculating expectation values:")
        for p in paulis:
            p_string = np.array(list(str(p)))
            counts[p] = self._get_pauli_expectation_value(p_string)
        return counts

    def _get_pauli_expectation_value(self, p_string: np.array) -> float:
        total_num = 0
        total_value = 0
        for basis in self.data.keys():
            if isinstance(basis, PauliBasis):
                basis_string = np.array(list(str(basis)))
            else:
                basis_string = np.array(basis)
            if np.alltrue(np.logical_or((basis_string == p_string), p_string == 'I')):
                basis_num, basis_value = self._get_value_and_times_for_basis(basis, p_string)
                total_num += basis_num
                total_value += basis_value
        return total_value / total_num

    def _get_value_and_times_for_basis(self, basis: np.array, p_string: np.array) -> Tuple[int, complex]:
        c = self.data[basis]
        basis_num = 0
        basis_value = 0
        for bit_string, times in c.items():
            value = self._get_value_from_bitstring(bit_string, p_string)
            basis_value += value * times
            basis_num += times
        return basis_num, basis_value

    @staticmethod
    def _get_value_from_bitstring(bit_string: str, p_string: np.array) -> complex:
        partial_bit_string = np.array(list(bit_string))[p_string != 'I']
        partial_bit_string = [int(ch) for ch in partial_bit_string]
        value = (-1) ** np.sum(partial_bit_string)
        return value

    def construct_constraint_matrix(self) -> ConstrainMatrix:
        """
        :return: construct the constraint matrix as defined in the main paper and
         stores it on self.constraint_matrix
        """
        constraint_matrix = ConstrainMatrix(self.sys_size, self.range_constraints, self.range_hamil_terms)
        paulis_needed_expectation = constraint_matrix.get_paulis_in_constraint_matrix()
        expectation_values = self.get_expectation_values_from_data(paulis_needed_expectation)
        constraint_matrix.get_constraint_matrix(expectation_values)
        self.constraint_matrix = constraint_matrix
        return constraint_matrix

    def update_ansatz_parameters_from_constrain_matrix(self) -> None:
        """
        updates the ansatz parameters according to the constraint matrix singular
        values as described in the main paper. Can be used only after
        self.construct_constraint_matrix was called. Note: all the constraint
        matrix singular values are converted into parameters, though only
        num_parameters are used for the construction of the Gibbs Hamiltonian.
        """
        terms = self.constraint_matrix.get_constraint_matrix_hamiltonians(
            num_hamiltonians=self.num_parameters)
        logging.info(f"Updating ansatz parameters from constraint matrix, num terms is {len(terms)}")
        self.update_ansatz_parameters(terms, [str(i) for i in range(len(terms))])

    def update_ansatz_parameters(self, parameters: List[np.array], names: Optional[List[str]] = None,
                                 max_coefficient_range: int = 1) -> None:
        """
        Update the ansatz's with new parameters with random coefficients with
        union distribution over [-max_coefficient_range,max_coefficient_range]

        :param parameters: list of new parameter coefficients ordered by the ansatz's
         parameters order.
        :param names: List of the new parameters names
        :param max_coefficient_range: the maximal (and minimal value) a random
         coefficient can get.
        :return:
        """
        if names is not None:
            for param, name in zip(parameters, names):
                if np.any([p.name == name for p in self.parameters]):
                    raise Exception(f"Name '{name}' already exists for parameter in ansatz")
                self.parameters.append(
                    Parameter(param, np.random.uniform() * 2 * max_coefficient_range - max_coefficient_range, name))
        else:
            for param in parameters:
                self.parameters.append(
                    Parameter(param, np.random.uniform() * 2 * max_coefficient_range - max_coefficient_range))

    @abc.abstractmethod
    def loss(self, params_coefficients: Optional[List[float]] = None):
        """
        Calculates the loss with new Parameters coefficients and update
        the ansatz coefficients with the new ones.

        :param params_coefficients: the coefficients of the parameters
         for which to calculate the loss with
        :return: the loss, chi squared, as defined in the main paper
        """
        ...


class SciPyAnsatz(Ansatz):
    non_lin_ls_bounds: float

    def __init__(self, sys_size: int, range_hamil_terms: int, range_constraints: int,
                 data: Dict[Union[PauliBasis, Tuple[str]], Counter], num_parameters: Optional[int] = None,
                 non_lin_ls_bounds: float = 100) -> None:
        """
        :param non_lin_ls_bounds: non linear least squares bound as defined
         in scipy.optimize.least_squares.
        """
        super().__init__(sys_size, range_hamil_terms, range_constraints, data, num_parameters)
        self.non_lin_ls_bounds = non_lin_ls_bounds

    def get_gibbs_hamiltonian_from_ansatz(self) -> np.array:
        """
        :return: the Gibbs Hamiltonian which the ansatz defines, as described
         in the main paper.
        """
        gibbs_hamiltonian = np.zeros_like(self.parameters[0].operator)
        for i, p in enumerate(self.parameters):
            if i <= self.num_parameters:
                gibbs_hamiltonian += p.get_parameter()
        return gibbs_hamiltonian

    def loss(self, params_coefficients: Optional[List[float]] = None) -> np.array:
        """
        Calculates the loss with new Parameters coefficients and update
        the ansatz coefficients with the new ones.

        :param params_coefficients: the coefficients of the parameters
         for which to calculate the loss with
        :return: the loss, chi squared, as defined in the main paper
        """
        return np.sum(np.square(self._residuals(params_coefficients))) / 2

    def _residuals(self, params_coefficients: Optional[List[float]] = None) -> List[float]:
        if params_coefficients is not None:
            for coefficient, p in zip(params_coefficients, self.parameters):
                p.coefficient = coefficient
        residuals = []
        for basis, counts in self.data.items():
            total_counts = sum(counts.values())
            if not isinstance(basis, PauliBasis):
                basis = PauliBasis(''.join(basis))
            u = qi.Operator(basis.basis_measurement(measure=False)).data
            rotated_forward = u @ self.forward() @ np.conjugate(np.transpose(u))
            for s in range(rotated_forward.shape[0]):
                s_bin_str = "{0:b}".format(s).zfill(self.sys_size)
                c = counts.get(s_bin_str, 0)
                p = c / total_counts
                residuals.append(p - np.real(rotated_forward[s, s]))
        return residuals

    def solve_iteratively(self, max_nfev: int = 30, jac: str = "2-point",
                          x0: Optional[np.array] = None) -> None:
        """
        Optimize the ansatz parameters according to the chi squared loss as
        described in the main paper. Optimization is done using
        scipy.optimize.least_squares function.

        :param max_nfev: max_nfev as defined in scipy.optimize.least_squares
        :param jac: jac as defined in max_nfev as defined in scipy.optimize.least_squares
        :param x0: x0 as defined in scipy.optimize.least_squares, if None, taken
         to be the parameters coefficients.
        :return:
        """
        if x0 is None:
            x0 = [x.coefficient for i, x in enumerate(self.parameters) if i < self.num_parameters]
        res = least_squares(self._residuals, x0, jac=jac, max_nfev=max_nfev, ftol=1e-3, loss='linear',
                            bounds=(-self.non_lin_ls_bounds if self.non_lin_ls_bounds is not None else -np.inf,
                                    self.non_lin_ls_bounds if self.non_lin_ls_bounds is not None else np.inf),
                            verbose=2)
        if res is not None:
            # Setting the optimal parameters
            self._residuals(res.x)
        logging.debug(f"Optimization result: {res}")


class TorchAnsatz(Ansatz, nn.Module):
    torch_parameters: nn.ParameterDict
    optimizer: Optional[torch.optim.Optimizer]
    lr: float

    def __init__(self, sys_size: int, range_hamil_terms: int, range_constraints: int,
                 data: Dict[Union[PauliBasis, Tuple[str]], Counter], num_parameters: Optional[Union[int, float]] = None,
                 lr=1, device: Union[str, torch.device] = 'cpu') -> None:
        """
        :param lr: initial learning rate for the Torch optimizer. Note that
         the learning rate is changed during the iterations using the Adam
         optimizer as defined in torch.optim.Adam.
        :param device: the device to do the Torch calculations on (CPU or GPU)
        """
        Ansatz.__init__(self, sys_size, range_hamil_terms, range_constraints, data, num_parameters)
        nn.Module.__init__(self)
        self.torch_parameters = nn.ParameterDict()
        self.optimizer = None
        self.lr = lr
        self.device = device
        self.to(self.device)

    def train_iteration(self) -> torch.Tensor:
        """
        Performs a single train iteration using PyTorch gradient decent.

        :return: loss after the iteration
        """
        self.optimizer.zero_grad()
        loss = self.loss()
        loss.backward()
        self.optimizer.step()
        for p in self.parameters:
            p.coefficient = self.torch_parameters[p.name].data
        return loss.item()

    def loss(self, params_coefficients: Optional[List[float]] = None) -> torch.Tensor:
        """
        Calculates the loss with new Parameters coefficients and update
        the ansatz coefficients with the new ones.

        :param params_coefficients: should be None, PyTorch ansatz does not
         support this function parameter.
        :return: the loss, chi squared, as defined in the main paper
        """
        if params_coefficients is not None:
            raise Exception("Can't update TorchAnsatz params manually")
        loss = torch.zeros(1).to(self.device)
        for basis, counts in self.data.items():
            total_counts = sum(counts.values())
            if not isinstance(basis, PauliBasis):
                basis = PauliBasis(''.join(basis))
            u = qi.Operator(basis.basis_measurement(measure=False)).data
            # noinspection PyTypeChecker
            rotated_forward = torch.from_numpy(u).to(self.device) @ self.forward().type(
                torch.cdouble) @ torch.from_numpy(np.conjugate(np.transpose(u))).to(self.device)
            for s in range(rotated_forward.shape[0]):
                s_bin_str = "{0:b}".format(s).zfill(self.sys_size)
                c = counts.get(s_bin_str, 0)
                p = c / total_counts
                loss += torch.pow(p - torch.real(rotated_forward[s, s]), 2)
        return loss

    def get_gibbs_hamiltonian_from_ansatz(self) -> torch.Tensor:
        """
        :return: the Gibbs Hamiltonian which the ansatz defines, as described
         in the main paper.
        """
        gibbs_hamiltonian = torch.zeros(self.parameters[0].operator.shape, dtype=torch.complex128,
                                        device=self.device)
        for i, p in enumerate(self.parameters):
            if i <= self.num_parameters:
                gibbs_hamiltonian += torch.from_numpy(p.operator).to(self.device) * self.torch_parameters[p.name]
        return gibbs_hamiltonian

    def update_ansatz_parameters(self, parameters: List[np.array], names: Optional[List[str]] = None,
                                 max_coefficient_range: int = 1):
        """
        Update the ansatz's with new parameters with random coefficients with
        union distribution over [-max_coefficient_range,max_coefficient_range]

        :param parameters: list of new parameter coefficients ordered by the ansatz's
         parameters order.
        :param names: List of the new parameters names
        :param max_coefficient_range: the maximal (and minimal value) a random
         coefficient can get.
        :return:
        """
        Ansatz.update_ansatz_parameters(self, parameters, names, max_coefficient_range)
        self._update_torch_parameters()
        self.optimizer = torch.optim.Adam(nn.Module.parameters(self), self.lr)

    def _update_torch_parameters(self):
        for p in self.parameters:
            self.torch_parameters[p.name] = nn.Parameter(torch.Tensor([p.coefficient]))
        self.torch_parameters.to(self.device)

    def train_ansatz(self, num_iterations: int = 1000) -> Ansatz:
        """
        Doing a full train of the TorchAnsatz using gradient decent and Adam
        optimizer

        :param num_iterations: number of iterations of gradient decent
        :return: the trained ansatz.
        """
        i = 0
        j = 0
        min_index = 0
        min_loss = float('inf')
        best_loss_parameters = None
        while j < num_iterations:
            loss = self.train_iteration()
            if loss < min_loss:
                min_index = j
                min_loss = loss
                best_loss_parameters = copy(self.__dict__)
                print(f"iteration {j} loss: {loss}")
            if j >= num_iterations:
                break
            i += 1
            j += 1
        logging.debug(f"Min loss iteration index is: {min_index}")
        self.__dict__ = best_loss_parameters
        return self


class AnsatzResults:
    cyclic_measurer: CyclicMeasurer = None
    ansatz: Ansatz = None
    loss: float = None
    reconstructed_density_matrix: np.array = None
    ground_truth_density_matrix: np.array = None
    fidelity: float = None

    def __init__(self,
                 cyclic_measurer: CyclicMeasurer = None,
                 ansatz: Ansatz = None,
                 loss: float = None,
                 ground_truth_density_matrix: np.array = None,
                 fidelity: float = None,
                 **kwargs
                 ) -> None:
        """
        :param cyclic_measurer: the cyclic measurer used for the measurements
        :param ansatz: the ansatz reconstructing the density matrix
        :param loss: the final loss after the optimization
        :param ground_truth_density_matrix: a ground truth density matrix to be
         compared with the ansatz result. Usually calculated using circuit
         simulation (with noise model in case of hardware backend).
        :param fidelity: fidelity between the reconstructed density matrix and
         ground_truth_density_matrix
        :param kwargs: other data to save.
        """
        self.backend_name = cyclic_measurer.backend.__str__()
        self.cyclic_measurer = cyclic_measurer
        self.ansatz = ansatz
        self.reconstructed_density_matrix = ansatz.forward()
        self.loss = loss
        self.ground_truth_density_matrix = ground_truth_density_matrix
        self.fidelity = fidelity
        self.backend_ver = cyclic_measurer.backend.properties().backend_version if \
            cyclic_measurer.backend.properties() is not None else None
        [setattr(self, k, v) for k, v in kwargs.items()]

    def save_results(self, directory: str) -> None:
        """
        Saves the results as a npz file. Note: this function uses pickle which
        may have issues for loading data after code changes. It is recommended
        to implement another method for saving results in order to reduce
        this dependency.

        :param directory: a global or relative path to save the results in
         with the following name format:
         backend_name_sys-size_num-parameters-time.npz
        """
        data = self.__dict__.copy()
        data['cyclic_measurer'].backend = Aer.get_backend('aer_simulator')
        time_str = time.strftime("%Y-%m-%d-%H_%M_%S")
        np.savez(f"{directory + self.backend_name}_{str(self.ansatz.sys_size)}_"
                 f"{str(self.ansatz.num_parameters)}_{time_str}", data)

    @classmethod
    def load_results(cls, name: str, directory: str):
        """
        :param name: the result file name (without the .npz extension)
        :param directory: a global or relative path to the directory where the
         results are saved.
        :return: the results object
        """
        data = np.load(f"{directory}/{name}.npz", allow_pickle=True)
        data = list(data.items())[0][1].item()
        ansatz = cls(**data)
        ansatz.__dict__ = data
        return ansatz


def get_ansatz(qubits_to_reconstruct: List[int], is_torch: bool, num_parameters: Optional[int], range_constraints: int,
               range_hamiltonian_terms: int, results: Dict[Union[PauliBasis, Tuple[str]], Counter]) -> Ansatz:
    """
    :param qubits_to_reconstruct: the qubits for which the density matrix will be
     reconstructed.
    :param is_torch: whether to use PyTorch (True) simulation or Scipy least squares
     gradient decent optimization (False).
    :param num_parameters: amount of parameters to used in the ansatz (defined as
     'l' in the main paper graphs).
    :param range_constraints: range of constraints in the constraint matrix as
     defined in the main paper
    :param range_hamiltonian_terms: range of Hamiltonian terms in the constraint matrix
     as defined in the main paper
    :param results: results dictionary for each Pauli basis
    :return: trained ansatz which reconstruct the density matrix as defined in
     the main paper algorithm.
    """
    if is_torch:
        ansatz = TorchAnsatz(len(qubits_to_reconstruct), range_hamiltonian_terms, range_constraints, results,
                             num_parameters=num_parameters)
    else:
        ansatz = SciPyAnsatz(len(qubits_to_reconstruct), range_hamiltonian_terms, range_constraints, results,
                             num_parameters=num_parameters)
    ansatz.construct_constraint_matrix()
    ansatz.update_ansatz_parameters_from_constrain_matrix()
    ansatz.solve_iteratively() if not is_torch else \
        ansatz.train_ansatz()
    return ansatz
