Authors: Tilas Kabengele

# helper functions for quantum computing

import numpy as np
import scipy as sp

def is_unitary(matrix):
    """
    Checks whether the given matrix is unitary or not.
    Returns True if the matrix is unitary, False otherwise.
    """
    # Check if the matrix is square
    n, m = matrix.shape
    if n != m:
        return False

    # Compute the conjugate transpose of the matrix
    conjugate_transpose = np.conj(matrix.T)

    # Compute the product of the matrix and its conjugate transpose
    product = np.dot(matrix, conjugate_transpose)

    # Check if the product is equal to the identity matrix within a tolerance
    identity = np.eye(n)
    return np.allclose(product, identity)

def is_hermitian(matrix):
    """
    Checks whether the given matrix is Hermitian or not.
    Returns True if the matrix is Hermitian, False otherwise.
    """
    # Check if the matrix is square
    n, m = matrix.shape
    if n != m:
        return False

    # Compute the conjugate transpose of the matrix
    conjugate_transpose = np.conj(matrix.T)

    # Check if the matrix is equal to its conjugate transpose within a tolerance
    return np.allclose(matrix, conjugate_transpose)

def TridiagA(lambda_func, mu_func, N):
    """
    Creates tridiagonal stochastic matrix for the Schlogl model
    Inputs lambda and mu model functions, and N desired size of matrix
    Returns stochastic matrix (non-hermitian)
    """
    # initialize diagonals
    d1 = np.zeros(N-1)
    d2 = np.zeros(N)
    d3 = np.zeros(N-1)

    # element 1,1
    d2[0] = -lambda_func(0)

    # element N,N
    d2[N-1] = -mu_func(N-1)

    # bottom diagonal elements
    for i in range(N-1):
        d1[i] = lambda_func(i)

    # top diagonal elements
    for i in range(1, N):
        d3[i-1] = mu_func(i)

    # main diagonal elements
    for i in range(1, N-1):
        d2[i] = -lambda_func(i) - mu_func(i)

    # putting the diagonals together
    Q = np.diag(d1, k=-1) + np.diag(d2, k=0) + np.diag(d3, k=1)
    return Q

def normalized_preconditioner(A):
    """
    Normalize a given matrix A.
    """
    A_normalized = A / np.linalg.norm(A, axis=0)
    return A_normalized

def matrix_to_unitary(M):
    # Compute the singular value decomposition of the matrix
    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    
    # Form a unitary matrix from the left singular vectors
    U_unitary = U * np.exp(1j * np.angle(np.linalg.det(U)))
    
    # Return the unitary matrix
    return U_unitary

def abs_smallest_values(arr):
    """
    Find the first and second smallest absolute values in array.
    Input: arr--> array, k--> number of smallest values
    """
    abs_vals = np.abs(arr)
    abs_vals_sorted = np.sort(abs_vals)
    return abs_vals_sorted

def find_eigenvalues_eigenvectors(matrix, n, m, hermitian):
    """
    Find the n-th and m-th eigenvalues and eigenvectors of a given matrix.

    Parameters:
    - matrix: input matrix
    - n, m: indices of the desired eigenvalues and eigenvectors
    - hermitian: boolean indicating whether the matrix is Hermitian

    Returns:
    - eigenvalue_n, eigenvector_n, eigenvalue_m, eigenvector_m
    """
    
    if not hermitian:
        # Find the eigenvalues and eigenvectors of the matrix
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
    else:
        # Find the eigenvalues and eigenvectors of the matrix using the appropriate method
        # For sparse matrices, use eigsh
        #eigenvalues, eigenvectors = sp.sparse.linalg.eigsh(matrix, k=max(m, n), which='LM')
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
            

    # Sort the eigenvalues and eigenvectors in ascending order by magnitude
    sorted_indices = np.argsort(np.abs(eigenvalues))
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Return the n-th and m-th eigenvalues and eigenvectors
    return eigenvalues[n - 1], eigenvectors[:, n - 1], eigenvalues[m - 1], eigenvectors[:, m - 1]

def smallest_singular_values(matrix, n):
    """
    Return the first n smallest magnitude singular values of a matrix.

    Parameters:
    - matrix: Input matrix
    - n: Number of singular values to return

    Returns:
    - singular_values: 1-D array containing the first n smallest magnitude singular values
    """

    # Compute SVD
    _, singular_values, _ = np.linalg.svd(matrix)

    # Sort singular values by magnitude
    sorted_indices = np.argsort(np.abs(singular_values))

    # Return the first n smallest magnitude singular values
    return singular_values[sorted_indices[:n]]

def random_hermitian(n):
    """
    Generates a random Hermitian matrix of size n x n.
    """
    # Generate a random complex matrix
    A = np.random.randn(n, n) 
    
    # Construct the Hermitian matrix by taking the sum of the matrix and its conjugate transpose
    H = 0.5*(A + np.conj(A.T))
    
    return H

from itertools import combinations
"""
Generates subsets of a given array. 
Does not include empty set or repititions, e.g., [1,2] and [2,1] are the same
"""
def generate_subsets(arr):
    subsets = [[]]  # Initialize with an empty subset
    for num in arr:
        new_subsets = []
        for subset in subsets:
            new_subset = subset + [num]
            new_subsets.append(new_subset)
        subsets.extend(new_subsets)
    subsets = [list(subset) for subset in subsets if subset]
    return subsets

"""
Generate sorted subarrays. Can be in ascending or descending order in terms of array size.
N: number of elements generated
subarray_maxsize: don't include subarrays greater than this
"""
def generate_subsets_sorted(arr, ascending=True, N=None, subarray_maxsize=None, subarray_minsize=None):
    subsets = [[]]  # Initialize with an empty subset
    for num in arr:
        subsets += [subset + [num] for subset in subsets]
        if N is not None and len(subsets) >= N:
            break
    subsets = [subset for subset in subsets if subset and (subarray_maxsize is None or len(subset) <= subarray_maxsize) and (subarray_minsize is None or len(subset) >= subarray_minsize)]
    subsets.sort(key=len, reverse=not ascending)  # Sort subsets based on length
    return subsets[:N]


"""
Given integer N, genrate array of elements, [0,1,..,N-1].
"""
def generate_indices(N):
    return list(range(N))

"""
Compute the number of possible combinations (subsets) of an array excluding empty set
This is known as a powerset and has a formula. Technically, a powerset includes empty set.
So, we subtract 1 from  the powerset.
combinations = 2**N - 1
N is the number of terms (elements)
"""
def count_combinations(N):
    return 2**N - 1


"""
extract closest N values to classical eigenvalues
"""
import math

def extract_closest_values(experimental_results, actual_value, N):
    differences = []
    for i, result in enumerate(experimental_results):
        if math.isnan(result):
            continue
        difference = abs(result - actual_value)
        differences.append((difference, result, i))

    differences.sort(key=lambda x: x[0])
    closest_values = []
    closest_indices = []
    count = 0
    for diff, value, index in differences:
        if math.isnan(value):
            continue
        closest_values.append(value)
        closest_indices.append(index)
        count += 1
        if count == N:
            break

    return closest_values, closest_indices

"""
Find common indices between two arrays
"""
def find_common_indices(array1, array2):
    common_indices = set(array1) & set(array2)
    return list(common_indices)

"""
replace unwanted state with zero
"""
def replace_digit_with_zero(array, digit):
    return np.where(array == digit, 0, array)


"""
Split counts and values from callbacks to extract the states we need
"""
import numpy as np

def split_counts_values(counts, values):
    state0_counts = []
    state1_counts = []
    state0 = []
    state1 = []

    if isinstance(counts, np.ndarray):
        counts = counts.tolist()  # Convert to Python list if input is NumPy array
        values = values.tolist()
        
    state1_start_index = counts.index(1, counts.index(1) + 1) if 1 in counts[counts.index(1) + 1:] else len(counts)

    state0_counts = counts[:state1_start_index]
    state1_counts = counts[state1_start_index:]

    state0 = values[:state1_start_index]
    state1 = values[state1_start_index:]

    return state0_counts, state1_counts, state0, state1


def commutator(A, B):
    return np.dot(A, B) - np.dot(B, A)


def pauliCommute(A, B):
    return A @ B - B @ A


def magnitude_sort(input_array):
    
    # Calculate the magnitude of complex numbers
    magnitudes = np.abs(input_array)

    # Get the indices that would sort the array by magnitude (largest to smallest)
    sorted_indices = np.argsort(magnitudes)[::-1]

    # Sort the original array using the sorted indices
    sorted_array = input_array[sorted_indices]
    return sorted_array, sorted_indices

def positive_first_sort(input_array):
    # Calculate the magnitude of complex numbers
    magnitudes = np.abs(input_array)

    # Separate positive and non-positive values
    positive_indices = np.where(input_array > 0)[0]
    non_positive_indices = np.where(input_array <= 0)[0]

    # Get the indices that would sort the positive values by magnitude (largest to smallest)
    sorted_positive_indices = positive_indices[np.argsort(magnitudes[positive_indices])[::-1]]

    # Get the indices that would sort the non-positive values by magnitude (smallest to largest)
    sorted_non_positive_indices = non_positive_indices[np.argsort(magnitudes[non_positive_indices])]

    # Concatenate the sorted indices
    sorted_indices = np.concatenate((sorted_positive_indices, sorted_non_positive_indices))

    # Sort the original array using the sorted indices
    sorted_array = input_array[sorted_indices]
    
    return sorted_array, sorted_indices


def paramVector2Array(param_vector_element_obj):
    # Determine the number of parameters in each set
    num_parameters_per_set = len(list(param_vector_element_obj[0]))

    # Initialize a list to store arrays corresponding to each set of parameters
    result_list_of_arrays = []

    # Iterate through each dictionary in the object
    for element_dict in param_vector_element_obj:
        # Extract the values (ignore the keys)
        values = list(element_dict.values())

        # Group the values into arrays corresponding to each set of parameters
        arrays = [values[i:i + num_parameters_per_set] for i in range(0, len(values), num_parameters_per_set)]

        # Add the arrays to the result list
        result_list_of_arrays.extend(arrays)

    return result_list_of_arrays


def nice_mat_print(matrix, decimal_places=3):
    # Determine the maximum width needed for each column
    column_widths = [max(len(f'{element:.{decimal_places}f}' if element.imag != 0 else f'{element.real:.{decimal_places}f}') for element in column) for column in matrix.T]

    for row in matrix:
        formatted_row = [f'{element:.{decimal_places}f}' if element.imag != 0 else f'{element.real:.{decimal_places}f}' for element in row]
        formatted_row = [element.rjust(width) for element, width in zip(formatted_row, column_widths)]
        print('  '.join(formatted_row))

def block_diagonal_matrix(Q):
    """
    Create a block diagonal square matrix in the form [0 Q; Q.T 0].
    
    Args:
        Q (numpy.ndarray): The square matrix to be placed in the center blocks.
        
    Returns:
        numpy.ndarray: The block diagonal square matrix.
    """
    n = Q.shape[0]
    
    top_left = np.zeros((n, n), dtype=Q.dtype)
    bottom_right = np.zeros((n, n), dtype=Q.dtype)

    Q_hermitian = np.block([[top_left, Q],
                            [Q.T, bottom_right]]
                           )

    return Q_hermitian

def get_n_min_magnitude_values(arr, n):
    if n <= 0:
        return [], []

    # Create a list of tuples containing the magnitude and the index of each element
    magnitude_index_pairs = [(abs(val), val, index) for index, val in enumerate(arr)]

    # Sort the list based on magnitude
    magnitude_index_pairs.sort()

    # Extract the first N values and their corresponding indices
    min_magnitude_values = [pair[1] for pair in magnitude_index_pairs[:n]]
    min_magnitude_indices = [pair[2] for pair in magnitude_index_pairs[:n]]

    return min_magnitude_values, min_magnitude_indices




