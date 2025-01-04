"""
Various Uuilities for simulating correlated data based on marginal
distributions of variables and Pearson correlations between variables.
"""

# Import required packages
import numpy as np
import math

# Read the csv files to pd data frams


def percentiles_to_probabilities(x_vec, perc_vec):
    """
    Converts a vector of percentile ranks into probabilities corresponding
    to a given support vector.

    Parameters:
        x_vec (array-like): A 1D array of support for the distribution.
        perc_vec (array-like): A 1D array of percentiles for each value in x_vec.
            Must be in the interval (0, 1).

    Returns:
        tuple: (sorted_x, probabilities)
            sorted_x (ndarray): Sorted version of x_vec.
            probabilities (ndarray): Probabilities corresponding to sorted_x.
    """
    # Ensure inputs are NumPy arrays
    x_vec = np.asarray(x_vec)
    perc_vec = np.asarray(perc_vec)


    #Convert nan to 0
    perc_vec = np.nan_to_num(perc_vec, nan = 0)

    # Input validation
    if x_vec.shape != perc_vec.shape:
        raise ValueError("x_vec and perc_vec must have the same shape.")
    if not ((0 <= perc_vec).all() and (perc_vec < 1).all()):
        raise ValueError("perc_vec must contain values strictly between 0 and 1.")


    # Sort by percentiles, descending
    sorted_indices = np.argsort(-perc_vec)
    sorted_x = x_vec[sorted_indices]
    sorted_perc = perc_vec[sorted_indices]

    probabilities = sorted_perc.copy()
    for i, p in enumerate(sorted_perc):
        if i == 0:
            probabilities[i] = (1 - p)
            continue
        else:
            probabilities[i] = (sorted_perc[i - 1] - p)

    # Ensure probabilities sum to 1
    if not np.isclose(np.sum(probabilities), 1):
        raise ValueError("Calculated probabilities do not sum to 1.")

    return sorted_x, probabilities

def corr_sim(sort_vec, to_order_vec, target_corr, tol_corr, max_iter):
    """
    Adjusts `to_order_vec` so its correlation with `sort_vec`
    approximates `target_corr` within a given tolerance.

    Parameters:
        sort_vec (np.ndarray): Reference vector for sorting.
        to_order_vec (np.ndarray): Vector to adjust.
        target_corr (float): Target correlation (0 to 1).
        tol_corr (float): Tolerance for achieving target correlation.
        max_iter (int): Maximum number of iterations.

    Returns:
        tuple:
            np.ndarray: Sorted version of `sort_vec`.
            np.ndarray: Adjusted version of `to_order_vec` aligned with the sorted `sort_vec`.
    """
    # Input validation
    if not isinstance(sort_vec, np.ndarray) or not isinstance(to_order_vec, np.ndarray):
        raise ValueError("sort_vec and to_order_vec must be NumPy arrays.")
    if sort_vec.shape != to_order_vec.shape:
        raise ValueError("sort_vec and to_order_vec must have the same shape.")
    if not (0 <= target_corr <= 1):
        raise ValueError("target_corr must be between 0 and 1.")
    if tol_corr <= 0:
        raise ValueError("tol_corr must be a positive number.")
    if max_iter <= 0:
        raise ValueError("max_iter must be a positive integer.")

    # Sort sort_vec to establish the reference ordering
    n_scores = len(sort_vec)
    master_vec = np.sort(sort_vec)
    using_vec = to_order_vec.copy()
    np.random.shuffle(using_vec)  # Randomize to start with uncorrelated state

    # Initial correlation
    corr = np.corrcoef(master_vec, using_vec)[0, 1]
    print(f"Initial correlation: {corr}")

    iterations = 0
    swaps = 0

    while abs(corr - target_corr) >= tol_corr and iterations < max_iter:
        # Determine the number of swaps dynamically
        num_swaps = max(1, math.ceil(n_scores * (abs(corr - target_corr) / 10)))

        for _ in range(num_swaps):
            # Randomly select two indices to swap
            idx1, idx2 = np.random.choice(n_scores, 2, replace=False)
            if corr < target_corr:
                # Increase correlation by swapping values that are misaligned with the master vector
                if (master_vec[idx1] < master_vec[idx2] and using_vec[idx1] > using_vec[idx2]) or \
                   (master_vec[idx1] > master_vec[idx2] and using_vec[idx1] < using_vec[idx2]):
                    using_vec[idx1], using_vec[idx2] = using_vec[idx2], using_vec[idx1]
            else:
                # Decrease correlation by swapping values that are aligned too closely
                if (master_vec[idx1] < master_vec[idx2] and using_vec[idx1] < using_vec[idx2]) or \
                   (master_vec[idx1] > master_vec[idx2] and using_vec[idx1] > using_vec[idx2]):
                    using_vec[idx1], using_vec[idx2] = using_vec[idx2], using_vec[idx1]

        # Update correlation and iteration count
        corr = np.corrcoef(master_vec, using_vec)[0, 1]
        iterations += 1
        swaps += num_swaps

    print(f"Final correlation: {corr} after {iterations} iterations and {swaps} swaps")
    return master_vec, using_vec

def multi_corr_sim(ordered_vec1, ordered_vec2, to_order_vec, target_corr1, target_corr2, tol_corr, max_iter):
    """
    Adjusts `to_order_vec` so its correlations with `ordered_vec1` and `ordered_vec2`
    approximate `target_corr1` and `target_corr2` within a given tolerance.

    Parameters:
        ordered_vec1 (np.ndarray): Reference vector 1 (fixed order).
        ordered_vec2 (np.ndarray): Reference vector 2 (fixed order).
        to_order_vec (np.ndarray): Vector to adjust.
        target_corr1 (float): Target correlation with `ordered_vec1`.
        target_corr2 (float): Target correlation with `ordered_vec2`.
        tol_corr (float): Tolerance for achieving target correlations.
        max_iter (int): Maximum number of iterations.

    Returns:
        np.ndarray: Adjusted version of `to_order_vec`.
    """
    # Input validation
    if not isinstance(ordered_vec1, np.ndarray) or not isinstance(ordered_vec2, np.ndarray) or not isinstance(to_order_vec, np.ndarray):
        raise ValueError("All input vectors must be NumPy arrays.")
    if ordered_vec1.shape != ordered_vec2.shape or ordered_vec1.shape != to_order_vec.shape:
        raise ValueError("All input vectors must have the same shape.")
    if not (0 <= target_corr1 <= 1 and 0 <= target_corr2 <= 1):
        raise ValueError("Target correlations must be between 0 and 1.")
    if tol_corr <= 0:
        raise ValueError("tol_corr must be a positive number.")
    if max_iter <= 0:
        raise ValueError("max_iter must be a positive integer.")

    # Initialize correlations
    adjusted_vec = to_order_vec.copy()
    corr1 = np.corrcoef(ordered_vec1, adjusted_vec)[0, 1]
    corr2 = np.corrcoef(ordered_vec2, adjusted_vec)[0, 1]
    print(f"Initial correlations: corr1 = {corr1}, corr2 = {corr2}")

    iterations = 0
    swaps = 0

    while (abs(corr1 - target_corr1) >= tol_corr or abs(corr2 - target_corr2) >= tol_corr) and iterations < max_iter:
        # Calculate deviations and probabilities for prioritization
        dev1 = abs(corr1 - target_corr1)
        dev2 = abs(corr2 - target_corr2)
        prob1 = dev1 / (dev1 + dev2) if (dev1 + dev2) > 0 else 0.5
        prob2 = 1 - prob1
        favor = np.random.choice([1, 2], p=[prob1, prob2])

        # Determine number of swaps dynamically
        num_swaps = max(1, int(len(to_order_vec) * 0.1 * max(dev1, dev2)))

        for _ in range(num_swaps):
            idx1, idx2 = np.random.choice(len(to_order_vec), 2, replace=False)

            if favor == 1:
                # Adjust correlation with ordered_vec1
                if ((adjusted_vec[idx1] < adjusted_vec[idx2]) != (ordered_vec1[idx1] < ordered_vec1[idx2]) and corr1 < target_corr1) or \
                   ((adjusted_vec[idx1] < adjusted_vec[idx2]) == (ordered_vec1[idx1] < ordered_vec1[idx2]) and corr1 > target_corr1):
                    adjusted_vec[idx1], adjusted_vec[idx2] = adjusted_vec[idx2], adjusted_vec[idx1]
            elif favor == 2:
                # Adjust correlation with ordered_vec2
                if ((adjusted_vec[idx1] < adjusted_vec[idx2]) != (ordered_vec2[idx1] < ordered_vec2[idx2]) and corr2 < target_corr2) or \
                   ((adjusted_vec[idx1] < adjusted_vec[idx2]) == (ordered_vec2[idx1] < ordered_vec2[idx2]) and corr2 > target_corr2):
                    adjusted_vec[idx1], adjusted_vec[idx2] = adjusted_vec[idx2], adjusted_vec[idx1]

        # Update correlations
        corr1 = np.corrcoef(ordered_vec1, adjusted_vec)[0, 1]
        corr2 = np.corrcoef(ordered_vec2, adjusted_vec)[0, 1]
        iterations += 1
        swaps += num_swaps

    print(f"Final correlations: corr1 = {corr1}, corr2 = {corr2} after {iterations} iterations and {swaps} swaps.")
    return adjusted_vec
