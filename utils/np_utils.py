import logging
from typing import Union, Optional

import numpy as np
import torch
from scipy.linalg import sqrtm, logm
# noinspection PyProtectedMember
from scipy.linalg._matfuncs_inv_ssq import LogmExactlySingularWarning
from torch import Tensor


def is_hermitian(A: np.array) -> bool:
    return np.allclose(np.conjugate(np.transpose(A)), A)


def get_fidelity(density_matrix: Union[Tensor, np.array], predicted_density_matrix: Union[Tensor, np.array]) -> float:
    predicted_sqrt = sqrtm(predicted_density_matrix, disp=True)
    if isinstance(density_matrix, torch.Tensor):
        density_matrix = density_matrix.detach().numpy()
    # noinspection PyUnresolvedReferences
    mul = predicted_sqrt @ density_matrix @ predicted_sqrt
    min_eig = min(np.linalg.eigh(mul)[0])
    sqrt = sqrtm(mul)
    if np.isnan(sqrt).any():
        logging.warning(f"Nan sqrt of matrix with eigenvalue matrix {min_eig}, adding minimal identity to get positive"
                        f"eigenvalues")
        mul -= np.identity(mul.shape[0]) * min_eig
        sqrt = sqrtm(mul)
    f = (np.trace(sqrt)) ** 2
    return np.real(f)


def get_gibbs_hamiltonian(density_matrix: np.array, eps=1e-10,
                          check_hermitian: bool = True,
                          step_size: float = 1e1, traceless: bool = True,
                          max_stabilization=1e-5) \
        -> Optional[np.array]:
    # Adding small identity for numeric stability
    if eps > max_stabilization:
        raise Exception("Stabilizing epsilon is too large for Gibbs Hamiltonian extraction")
    density_matrix = density_matrix + eps * np.identity(density_matrix.shape[0])
    try:
        gibbs_hamiltonian = -np.array(logm(density_matrix))
        if not is_hermitian(gibbs_hamiltonian):
            if check_hermitian:
                logging.warning(f"Gibbs Hamiltonian is not hermitian, increasing eps to {eps * step_size}")
                return get_gibbs_hamiltonian(density_matrix, eps=eps * step_size, check_hermitian=check_hermitian)
            else:
                logging.warning("Gibbs Hamiltonian not hermitian")
    except LogmExactlySingularWarning:
        logging.warning(f"Singularity in Gibbs Hamiltonian, increasing eps to {eps * step_size}")
        return get_gibbs_hamiltonian(density_matrix, eps=eps * step_size, check_hermitian=check_hermitian)
    if np.isnan(gibbs_hamiltonian).any():
        logging.warning(f"Nan in Gibbs Hamiltonian, increasing eps to {eps * step_size}")
        return get_gibbs_hamiltonian(density_matrix, eps=eps * step_size, check_hermitian=check_hermitian)
    if np.max(np.linalg.eigh(gibbs_hamiltonian)[0]) >= 1e7:
        logging.warning(f"Gibbs Hamiltonian has eigenvalue smaller than -1e7, increasing eps to {eps * step_size}")
        return get_gibbs_hamiltonian(density_matrix, eps=eps * step_size, check_hermitian=check_hermitian)
    if traceless:
        n = gibbs_hamiltonian.shape[0]
        gibbs_hamiltonian -= np.identity(n) * (1 / 2 ** n) * np.trace(gibbs_hamiltonian)
    return gibbs_hamiltonian


def normalized_matrix(m: Union[np.array, torch.Tensor]):
    if isinstance(m, torch.Tensor):
        return m / torch.trace(m)
    else:
        return m / np.trace(m)
