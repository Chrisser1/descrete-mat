import math
import re as regex
from itertools import combinations, permutations

import pandas as pd
from sympy import *
from sympy.ntheory.modular import crt
from math import comb
init_printing()

def superscriptify(expr):
    """Convert an expression's powers (**) into unicode superscripts and leave multiplication as is."""
    if expr is None:
        return "*"
    s = str(expr)

    # Map digits and minus sign to superscripts
    superscript_map = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
        '-': '⁻'
    }

    def replace_power(match):
        exponent_str = match.group(1)
        # Convert each character of the exponent to its superscript equivalent
        return ''.join(superscript_map.get(ch, ch) for ch in exponent_str)

    # Replace all occurrences of '**<integer>' with the corresponding superscript characters
    s = regex.sub(r'\*\*(-?\d+)', lambda m: replace_power(m), s)

    return s


'''
    EUCLIDEAN ALGORITHM FUNCTIONS
'''

def extended_euclidean_algorithm_polys(N: Poly, M: Poly, x: Symbol, beatify: bool = true):
    # Initialize variables for the extended Euclidean algorithm
    r = [N, M]  # Remainders
    s = [Poly(1, x), Poly(0, x)]  # Coefficients of N(x)
    t = [Poly(0, x), Poly(1, x)]  # Coefficients of M(x)

    # Perform the extended Euclidean algorithm
    k = 0
    while r[k+1] != 0:
        quotient, remainder = div(r[k], r[k+1])
        r.append(remainder)
        if remainder == 0:
            s.append(None)
            t.append(None)
        else:
            s.append(s[k] - quotient * s[k+1])
            t.append(t[k] - quotient * t[k+1])
        k += 1

    results = []
    for i in range(len(r)):
        if beatify:
            r_str = superscriptify(r[i].as_expr() if r[i] is not None else None)
            s_str = superscriptify(s[i].as_expr() if s[i] is not None else None)
            t_str = superscriptify(t[i].as_expr() if t[i] is not None else None)
        else:
            r_str = r[i].as_expr() if r[i] is not None else None
            s_str = s[i].as_expr() if s[i] is not None else None
            t_str = t[i].as_expr() if t[i] is not None else None
        results.append({
            "k": i,
            "r_k(x)": r_str,
            "s_k(x)": s_str,
            "t_k(x)": t_str
        })

    df = pd.DataFrame(results)
    return df

def extended_euclidean_algorithm_integers(a, b, beautify: bool = True):
    """
    Compute the Extended Euclidean Algorithm steps for two integers a and b.

    Returns a DataFrame with columns:
    k, r_k, s_k, t_k
    such that r_k = s_k*a + t_k*b at each step.
    """
    # Initializations
    r = [a, b]
    s = [1, 0]
    t = [0, 1]

    k = 0
    while r[k+1] != 0:
        quotient = r[k] // r[k+1]
        remainder = r[k] % r[k+1]
        r.append(remainder)

        if remainder == 0:
            s.append(None)
            t.append(None)
        else:
            s.append(s[k] - quotient * s[k+1])
            t.append(t[k] - quotient * t[k+1])
        k += 1

    results = []
    for i in range(len(r)):
        if beautify:
            r_str = superscriptify(r[i]) if r[i] is not None else None
            s_str = superscriptify(s[i]) if s[i] is not None else None
            t_str = superscriptify(t[i]) if t[i] is not None else None
        else:
            r_str = r[i]
            s_str = s[i]
            t_str = t[i]
        results.append({
            "k": i,
            "r_k": r_str,
            "s_k": s_str,
            "t_k": t_str
        })

    df = pd.DataFrame(results)
    return df

class PermutationFilter:
    def __init__(self, sequence):
        self.sequence = sequence
        # Generate all permutations of the given sequence as tuples
        self.all_permutations = list(permutations(self.sequence))

    def filter_permutations(self, include_substrings=None, exclude_substrings=None):
        """
        Filter permutations based on optional inclusion and exclusion criteria.

        :param include_substrings: A list of substrings. A permutation must contain
                                   at least one of these substrings to pass.
        :param exclude_substrings: A list of substrings. A permutation must not contain
                                   any of these substrings to pass.

        :return: A list of permutations (as tuples) that match the criteria.
        """
        filtered = self.all_permutations

        # Convert tuple permutations into strings for checking
        # We'll do this step once to avoid repeating it for each condition
        filtered_strings = [''.join(p) for p in filtered]

        # If include_substrings is given, keep only those permutations that contain
        # at least one of the given substrings
        if include_substrings:
            new_filtered = []
            new_filtered_strings = []
            for perm_str, perm_tuple in zip(filtered_strings, filtered):
                if any(sub in perm_str for sub in include_substrings):
                    new_filtered.append(perm_tuple)
                    new_filtered_strings.append(perm_str)
            filtered = new_filtered
            filtered_strings = new_filtered_strings

        # If exclude_substrings is given, remove any permutations that contain
        # any of the given substrings
        if exclude_substrings:
            new_filtered = []
            new_filtered_strings = []
            for perm_str, perm_tuple in zip(filtered_strings, filtered):
                if not any(sub in perm_str for sub in exclude_substrings):
                    new_filtered.append(perm_tuple)
                    new_filtered_strings.append(perm_str)
            filtered = new_filtered
            filtered_strings = new_filtered_strings

        return filtered

def derangement_formula(n):
    """ Calculate the number of derangements using the direct formula. """
    derangements = math.factorial(n) * sum(((-1) ** k) / math.factorial(k) for k in range(n + 1))
    return int(derangements)

def count_edges_in_n_cube(n):
    # Number of edges in the n-cube graph
    return n * (2 ** (n - 1))

def guaranteed_selection_count(target, numbers):
    # Step 1: Find all subsets whose sum equals the target
    target_subsets = []
    for r in range(1, len(numbers) + 1):
        for subset in combinations(numbers, r):
            if sum(subset) == target:
                target_subsets.append(set(subset))

    # If no subset sums to the target, no finite number will guarantee it.
    if not target_subsets:
        return float('inf')  # or len(numbers) + 1, depending on your preference

    # Step 2: Find the maximum subset of 'numbers' that does NOT contain any target-sum subset fully
    # We'll brute force all subsets of 'numbers' and check.
    # For each candidate subset, we must ensure it does not include any of the target_subsets fully.
    # i.e., For every target_subset T, candidate_subset should not be a superset of T.

    max_safe_size = 0
    n = len(numbers)
    for r in range(n + 1):
        for candidate in combinations(numbers, r):
            candidate_set = set(candidate)
            # Check if candidate_set fully contains any target_subset
            if any(t_subset.issubset(candidate_set) for t_subset in target_subsets):
                # This candidate is not safe because it contains a full target-sum subset
                continue
            # If it passes all checks, update max_safe_size
            max_safe_size = max(max_safe_size, r)

    # Step 3: Once you exceed max_safe_size by 1, you guarantee forming a target subset
    return max_safe_size + 1

def check_relation_properties(A, R):
    # Convert R to a set for faster membership testing if needed
    R_set = set(R)

    # Check Reflexive: (a,a) in R for all a in A
    reflexive = all((a, a) in R_set for a in A)

    # Check Symmetric: For every (a,b) in R, (b,a) must also be in R
    symmetric = all((b, a) in R_set for (a, b) in R_set)

    # Check Antisymmetric: If (a,b) and (b,a) in R, then a must be b
    # If we ever find a != b with both (a,b) and (b,a), it's not antisymmetric.
    antisymmetric = True
    for (a, b) in R_set:
        if a != b and (b, a) in R_set:
            antisymmetric = False
            break

    # Check Transitive: If (a,b) and (b,c) in R, then (a,c) must be in R
    # We'll use a nested loop approach.
    transitive = True
    for (a, b) in R_set:
        for (x, y) in R_set:
            if b == x and (a, y) not in R_set:
                transitive = False
                break
        if not transitive:
            break

    # Return a dictionary or tuple of results
    return {
        "reflexive": reflexive,
        "symmetric": symmetric,
        "antisymmetric": antisymmetric,
        "transitive": transitive
    }

def test_formula(formula_func, reference_func, test_values=range(1, 11)):
    """
    Test a given formula function against a given reference function for a range of test values.
    Returns True if formula_func(n) == reference_func(n) for all tested values of n, False otherwise.
    """
    for n in test_values:
        lhs = formula_func(n)
        rhs = reference_func(n)
        if lhs != rhs:
            return False
    return True

def solve_linear_congruence(a, b, m):
    g = gcd(a, m)
    # Tjek om løsning eksisterer
    if b % g != 0:
        return None

    # Reducer
    a_red = a // g
    b_red = b // g
    m_red = m // g

    a_inv = mod_inverse(a_red, m_red)
    x0 = (b_red * a_inv) % m_red

    return (x0, m_red, g)

def solve_system_congruences(equations):
    """
    Solve a system of linear congruences given by:
    equations = [(A1, B1, M1), (A2, B2, M2), ...]
    meaning A_i * x ≡ B_i (mod M_i).

    Returns: (solution, modulus) if a solution exists, otherwise None.
    """
    moduli = []
    remainders = []

    for (A, B, M) in equations:
        g = gcd(A, M)
        # Check if solution exists for this congruence
        if B % g != 0:
            return None  # No solution

        # Reduce the congruence
        A_red = A // g
        B_red = B // g
        M_red = M // g

        # Compute the inverse of A_red mod M_red
        A_inv = mod_inverse(A_red, M_red)
        # Compute a particular solution x0
        x0 = (B_red * A_inv) % M_red

        moduli.append(M_red)
        remainders.append(x0)

    # Now solve the standard form system x ≡ remainders[i] (mod moduli[i])
    solution, modulus = crt(moduli, remainders)
    return (solution, modulus)
