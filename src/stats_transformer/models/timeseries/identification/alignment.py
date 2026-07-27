import numpy as np
from itertools import permutations

def align_signs(impact_matrix):
    """
    Enforces a positive main diagonal on the structural impact matrix.
    Since changing the sign of a column corresponds to multiplying the shock by -1,
    this is a valid normalization that resolves sign indeterminacy.
    
    Args:
        impact_matrix (np.ndarray): The (K, K) structural impact matrix.
        
    Returns:
        np.ndarray: The sign-aligned impact matrix.
    """
    aligned = impact_matrix.copy()
    for i in range(aligned.shape[1]):
        if aligned[i, i] < 0:
            aligned[:, i] = -aligned[:, i]
    return aligned

def align_permutation_to_target(impact_matrix, target_matrix):
    """
    Reorders the columns of the impact matrix to match a target matrix as closely as possible.
    Matches the columns based on maximizing the trace of (target_matrix^{-1} @ impact_matrix)
    or by minimizing the Frobenius norm of the difference.
    We minimize the Frobenius norm across all permutations.
    
    Args:
        impact_matrix (np.ndarray): The (K, K) structural impact matrix.
        target_matrix (np.ndarray): The (K, K) target matrix.
        
    Returns:
        np.ndarray: The permutation-aligned impact matrix.
    """
    K = impact_matrix.shape[1]
    best_perm = None
    min_dist = np.inf
    
    for perm in permutations(range(K)):
        permuted_matrix = impact_matrix[:, list(perm)]
        # We also need to align signs after permutation to compute distance fairly
        permuted_matrix = align_signs(permuted_matrix)
        target_aligned = align_signs(target_matrix)
        
        dist = np.linalg.norm(permuted_matrix - target_aligned, ord='fro')
        if dist < min_dist:
            min_dist = dist
            best_perm = list(perm)
            
    best_matrix = impact_matrix[:, best_perm]
    return align_signs(best_matrix)

def align_to_cholesky(impact_matrix, p_chol):
    """
    Aligns the impact matrix to the lower-triangular Cholesky factor of the residual covariance.
    This resolves permutation indeterminacy by finding the ordering closest to a recursive ordering.
    
    Args:
        impact_matrix (np.ndarray): The (K, K) structural impact matrix.
        p_chol (np.ndarray): The (K, K) lower-triangular Cholesky factor.
        
    Returns:
        np.ndarray: The aligned impact matrix.
    """
    return align_permutation_to_target(impact_matrix, p_chol)
