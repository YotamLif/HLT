import itertools
import logging
from abc import abstractmethod, ABC
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Union, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile, execute
from qiskit.providers import Backend
from qiskit.providers.aer import AerSimulator
from qiskit.result import Result


@dataclass
class Basis(ABC):
    basis: str

    def __init__(self, basis: str) -> None:
        self.basis = basis

    def __str__(self) -> str:
        return self.basis

    def __hash__(self) -> int:
        return hash(self.__str__())

    @abstractmethod
    def basis_measurement(self, measure: bool = True):
        pass


@dataclass
class PauliBasis(Basis):

    def __init__(self, basis: str) -> None:
        super().__init__(basis)

    def __hash__(self):
        return super.__hash__(self)

    def basis_measurement(self, measure: bool = True) -> QuantumCircuit:
        basis = self.__str__()
        n_qubits = len(basis)
        circ = QuantumCircuit(n_qubits)
        for i, b in enumerate(reversed(basis)):
            if b == 'X':
                circ.h(i)
            elif b == 'Y':
                circ.sdg(i)
                circ.h(i)
            elif b == 'Z':
                pass
            else:
                raise Exception('Bad measurement basis')
        if measure:
            circ.measure_all()
        return circ


class CyclicMeasurer:
    bases = ['X', 'Y', 'Z']

    qubits: List[int]
    system_size: int
    circ: QuantumCircuit
    backend: Optional[AerSimulator]

    def __init__(self, qubits: List[int], circ: QuantumCircuit,
                 backend: Optional[Backend] = None) -> None:
        self.qubits = qubits
        self.system_size = len(qubits)
        self.circ = circ
        if circ.num_clbits != 0:
            raise ValueError("Analysis can be done only on a circuit without any classical bits")
        self.backend = backend
        self.circs = None

    def get_cyclic_measurements_results(self, cycle_size: int, total_number_of_shots: int,
                                        optimization_level: int = 3,
                                        initial_layout: List[int] = None) \
            -> Dict[PauliBasis, Counter]:
        cycle_size = self._get_cycle_size(cycle_size)
        bases = self.get_pauli_bases(cycle_size)
        shots_per_experiment = self.get_number_of_shots_per_experiment(bases, total_number_of_shots)
        circs = []
        for b in bases:
            transpiled_circ = self._get_circ_with_measurement(b, initial_layout, optimization_level)
            circs.append((b, transpiled_circ))
        self.circs = circs
        results = execute([circ[1] for circ in circs], self.backend, shots=shots_per_experiment).result()
        results_dict = self._get_dict_from_results(circs, results)
        return results_dict

    def _get_circ_with_measurement(self, b: PauliBasis, initial_layout: List[int], optimization_level: int) -> \
            QuantumCircuit:
        circ = self._get_circuit_with_classical_bits()
        circ.append(b.basis_measurement().to_instruction(), self.qubits, range(len(self.qubits)))
        transpiled_circ = transpile(circ, self.backend, optimization_level=optimization_level,
                                    initial_layout=initial_layout)
        return transpiled_circ

    def _get_circuit_with_classical_bits(self) -> QuantumCircuit:
        circ = QuantumCircuit(len(self.circ.qubits), self.system_size)
        if self.circ.num_clbits != 0:
            raise Exception('Circ must not have classical bits!')
        circ.append(self.circ.to_instruction(), range(len(self.circ.qubits)))
        return circ

    @staticmethod
    def get_cycle_size(range_hamiltonians: int, range_constraints: int) -> int:
        return range_hamiltonians + range_constraints - 1

    @staticmethod
    def _get_dict_from_results(circs: List[Tuple[PauliBasis, QuantumCircuit]], results: Result) \
            -> Dict[PauliBasis, Counter]:
        results_dict = defaultdict(lambda: Counter())
        for b, circ in circs:
            counts = results.get_counts(circ)
            results_dict[b].update(counts)
        # Can't pickle default dict with lambda
        results_dict = dict(results_dict)
        return results_dict

    def get_pauli_bases(self, cycle_size: int) -> List[PauliBasis]:
        bases = self.get_pauli_labels(cycle_size)
        bases = [PauliBasis(''.join(basis)) for basis in bases]
        return bases

    def get_pauli_labels(self, cycle_size) -> List[Tuple[str]]:
        cycle_bases = itertools.product(self.bases, repeat=cycle_size)
        num_cycles = int(np.ceil(self.system_size / float(cycle_size)))
        bases = [(basis * num_cycles)[:self.system_size] for basis in cycle_bases]
        return bases

    @staticmethod
    def get_number_of_shots_per_experiment(bases: List[Union[tuple, PauliBasis]], total_number_of_shots: int) -> int:
        shots_per_experiment = total_number_of_shots // len(bases)
        logging.debug(f"Total number of measurements for cyclic measure is: {shots_per_experiment * len(bases)}")
        return shots_per_experiment

    def _get_cycle_size(self, cycle_size: int):
        if cycle_size > self.system_size:
            logging.warning(f"Cycle size ({cycle_size}) greater than sys size ({self.system_size}), "
                            f"will use cycles system size cycles")
            cycle_size = self.system_size
        return cycle_size
