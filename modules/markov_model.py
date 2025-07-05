from collections import defaultdict

# ------------------------------
# Build a normalized transition matrix from observed state sequence
# ------------------------------
def build_transition_matrix(states, state_order=None):
    """
    Constructs a transition probability matrix from a sequence of observed states.
    
    Args:
        states (list): List of observed states (e.g., ['Low', 'High', 'Low', ...])
        state_order (list, optional): Specific order of unique states. If None, sorted set is used.
    
    Returns:
        matrix (list of lists): Transition matrix where matrix[i][j] = P(j | i)
        state_order (list): Order of states corresponding to matrix indices
    """
    if state_order is None:
        state_order = sorted(set(states))  # Automatically determine state order

    state_idx = {s: i for i, s in enumerate(state_order)}  # Map state to index
    n = len(state_order)
    matrix = [[0 for _ in range(n)] for _ in range(n)]  # Initialize zero matrix

    # Count transitions from state a → b
    for a, b in zip(states[:-1], states[1:]):
        i, j = state_idx[a], state_idx[b]
        matrix[i][j] += 1

    # Normalize each row to convert to probabilities
    for i in range(n):
        row_sum = sum(matrix[i])
        if row_sum == 0:
            matrix[i][i] = 1  # Handle no transitions by self-loop
        else:
            matrix[i] = [v / row_sum for v in matrix[i]]

    return matrix, state_order

# ------------------------------
# Compute steady-state distribution using iterative method
# ------------------------------
def compute_steady_state(matrix, tol=1e-8, max_iter=1000):
    """
    Calculates the steady-state distribution π such that πP = π.

    Args:
        matrix (list of lists): Transition matrix
        tol (float): Convergence threshold
        max_iter (int): Maximum number of iterations

    Returns:
        dict: Mapping index → steady-state probability
    """
    n = len(matrix)
    prob = [1/n] * n  # Start with uniform distribution

    for _ in range(max_iter):
        new_prob = [0] * n
        for j in range(n):
            new_prob[j] = sum(prob[i] * matrix[i][j] for i in range(n))
        diff = sum(abs(new_prob[i] - prob[i]) for i in range(n))
        prob = new_prob
        if diff < tol:
            break  # Converged

    return {i: round(p, 6) for i, p in enumerate(prob)}

# ------------------------------
# Compute Mean First Recurrence Time for each state
# ------------------------------
def compute_recurrence_times(steady_probs, state_order):
    """
    Calculates average recurrence time for each state: R_i = 1 / π_i

    Args:
        steady_probs (dict): Mapping index → steady-state prob
        state_order (list): Mapping of indices to state names

    Returns:
        dict: Mapping state name → recurrence time
    """
    return {
        state_order[i]: round(1 / prob, 4)
        for i, prob in steady_probs.items()
    }

# ------------------------------
# Gaussian elimination solver for linear equations (Ax = b)
# ------------------------------
def solve_linear(A, b):
    """
    Solves Ax = b using Gaussian elimination without any external library.

    Args:
        A (list of lists): Coefficient matrix
        b (list): Constant vector

    Returns:
        list: Solution vector x
    """
    n = len(A)
    for i in range(n):
        # Pivot if diagonal element is zero
        if A[i][i] == 0:
            for j in range(i+1, n):
                if A[j][i] != 0:
                    A[i], A[j] = A[j], A[i]
                    b[i], b[j] = b[j], b[i]
                    break

        # Normalize pivot row
        factor = A[i][i]
        A[i] = [a / factor for a in A[i]]
        b[i] /= factor

        # Eliminate below
        for j in range(i+1, n):
            factor = A[j][i]
            A[j] = [a - factor * A[i][k] for k, a in enumerate(A[j])]
            b[j] -= factor * b[i]

    # Back-substitution
    x = [0] * n
    for i in reversed(range(n)):
        x[i] = b[i] - sum(A[i][j] * x[j] for j in range(i+1, n))

    return x

# ------------------------------
# Compute First Passage Time between states
# ------------------------------
def compute_first_passage(matrix, state_order):
    """
    Computes expected steps from state i to j (i ≠ j) using linear equations.

    Args:
        matrix (list of lists): Transition matrix
        state_order (list): Mapping of indices to state names

    Returns:
        dict: Nested dictionary mapping state_i → state_j → E[steps]
    """
    n = len(state_order)
    passage = {}

    for j in range(n):  # Target state
        for i in range(n):  # Start state
            if i == j:
                continue

            A, b = [], []
            for k in range(n):
                row = []
                if k == j:
                    # Set equation: x_j = 0
                    row = [0] * n
                    row[k] = 1
                    b.append(0)
                else:
                    for l in range(n):
                        if k == l:
                            row.append(1)
                        elif l == j:
                            row.append(0)
                        else:
                            row.append(-matrix[k][l])
                    b.append(1)
                A.append(row)

            x = solve_linear([r[:] for r in A], b[:])
            if state_order[i] not in passage:
                passage[state_order[i]] = {}
            passage[state_order[i]][state_order[j]] = round(x[i], 4)

    return passage

# ------------------------------
# Compute Absorption Times for transient → absorbing states
# ------------------------------
def compute_absorption(matrix, state_order):
    """
    Computes expected absorption time for transient states into absorbing ones.
    Assumes absorbing states have P[i][i] = 1 and no outgoing transitions.

    Args:
        matrix (list of lists): Transition matrix
        state_order (list): State labels

    Returns:
        dict: State → expected absorption time (if any absorbing states exist)
    """
    n = len(state_order)
    absorbing = [i for i in range(n) if matrix[i][i] == 1.0]
    transient = [i for i in range(n) if i not in absorbing]

    if not absorbing or not transient:
        return None  # No absorbing states present

    # Build Q matrix (transient-to-transient)
    Q = [[matrix[i][j] for j in transient] for i in transient]
    
    # Compute I - Q
    I_minus_Q = [
        [(1 if i == j else 0) - Q[i][j] for j in range(len(Q))]
        for i in range(len(Q))
    ]

    b = [1] * len(Q)  # RHS for time equations
    t_vals = solve_linear(I_minus_Q, b)

    return {
        state_order[transient[i]]: round(t_vals[i], 4)
        for i in range(len(t_vals))
    }
