# ------------------------------
# Forward Algorithm
# ------------------------------
def forward_algorithm(obs_seq, states, start_prob, trans_prob, emit_prob):
    """
    Computes the probability of the observation sequence given the model (P(O | λ)) using the Forward Algorithm.

    Args:
        obs_seq (list): Sequence of observed emissions (e.g., ['High', 'Moderate', 'Low'])
        states (list): List of hidden states
        start_prob (dict): Initial probabilities for each state
        trans_prob (dict of dict): Transition probability from state i to state j
        emit_prob (dict of dict): Emission probability of observation o from state s

    Returns:
        float: Total probability of the observation sequence
    """
    T = len(obs_seq)
    N = len(states)

    # fwd[t][i] = P(O1...Ot, Qt=state_i)
    fwd = [[0.0 for _ in range(N)] for _ in range(T)]

    # Initialization step (t = 0)
    for i in range(N):
        fwd[0][i] = start_prob[states[i]] * emit_prob[states[i]].get(obs_seq[0], 1e-6)  # Use small default if missing

    # Recursion step
    for t in range(1, T):
        for j in range(N):
            total = 0.0
            for i in range(N):
                total += fwd[t - 1][i] * trans_prob[states[i]][states[j]]
            fwd[t][j] = total * emit_prob[states[j]].get(obs_seq[t], 1e-6)

    # Termination step: sum of final step probabilities
    return round(sum(fwd[T - 1]), 6)

# ------------------------------
# Viterbi Algorithm
# ------------------------------
def viterbi_algorithm(obs_seq, states, start_prob, trans_prob, emit_prob):
    """
    Finds the most likely sequence of hidden states given observations using the Viterbi algorithm.

    Args:
        obs_seq (list): Observed emission sequence
        states (list): Possible hidden states
        start_prob (dict): Starting probability of each state
        trans_prob (dict of dict): State transition probabilities
        emit_prob (dict of dict): Emission probabilities for each observation given a state

    Returns:
        list: Most probable path of hidden states
    """
    T = len(obs_seq)
    N = len(states)

    # viterbi[t][i]: max probability of any path ending in state_i at time t
    viterbi = [[0.0] * N for _ in range(T)]
    backpointer = [[0] * N for _ in range(T)]  # To trace back the path

    # Initialization (t = 0)
    for i in range(N):
        viterbi[0][i] = start_prob[states[i]] * emit_prob[states[i]].get(obs_seq[0], 1e-6)

    # Recursion
    for t in range(1, T):
        for j in range(N):
            max_prob = 0.0
            max_index = 0
            for i in range(N):
                prob = viterbi[t - 1][i] * trans_prob[states[i]][states[j]]
                if prob > max_prob:
                    max_prob = prob
                    max_index = i
            viterbi[t][j] = max_prob * emit_prob[states[j]].get(obs_seq[t], 1e-6)
            backpointer[t][j] = max_index

    # Backtrace: find last state with max prob
    last_state = max(range(N), key=lambda i: viterbi[T - 1][i])
    best_path = [last_state]

    # Reconstruct the full path backward
    for t in range(T - 1, 0, -1):
        best_path.insert(0, backpointer[t][best_path[0]])

    return [states[i] for i in best_path]

# ------------------------------
# Compute Hidden-State Steady-State Distribution
# ------------------------------
def compute_hidden_steady_state(states, trans_prob, max_iter=1000, tol=1e-8):
    """
    Computes the steady-state distribution over hidden states using iterative convergence.

    Args:
        states (list): List of hidden states
        trans_prob (dict of dict): Transition probabilities between states
        max_iter (int): Maximum number of iterations to run
        tol (float): Convergence threshold

    Returns:
        dict: Mapping of state name → steady-state probability
    """
    num_states = len(states)
    pi = [1.0 / num_states] * num_states  # Start with uniform distribution

    for _ in range(max_iter):
        new_pi = [0.0] * num_states
        for j in range(num_states):
            for i in range(num_states):
                new_pi[j] += pi[i] * trans_prob[states[i]][states[j]]

        # Normalize probabilities
        total = sum(new_pi)
        new_pi = [x / total for x in new_pi]

        # Check convergence
        if max(abs(new_pi[i] - pi[i]) for i in range(num_states)) < tol:
            break
        pi = new_pi

    return {states[i]: pi[i] for i in range(num_states)}
