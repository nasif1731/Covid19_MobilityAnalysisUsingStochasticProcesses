def mm1_metrics(arrival_rate, service_rate):
    """
    Computes standard performance metrics for an M/M/1 queuing system.
    
    Args:
        arrival_rate (float): λ, the average arrival rate (e.g., customers per minute)
        service_rate (float): μ, the average service rate (e.g., customers served per minute)

    Returns:
        dict: A dictionary containing key M/M/1 queue metrics:
              - utilization (ρ)
              - average number in system (L)
              - average number in queue (Lq)
              - average time in system (W)
              - average waiting time in queue (Wq)
              - probability system is empty (P0)

    Raises:
        ValueError: If the system is unstable (arrival_rate >= service_rate)
    """

    # ⚠️ Check system stability (λ must be less than μ)
    if arrival_rate >= service_rate:
        raise ValueError("System unstable: arrival rate must be less than service rate (λ < μ).")

    # ρ (rho): Utilization factor — how busy the server is
    rho = arrival_rate / service_rate

    # L: Average number of customers in the system
    L = rho / (1 - rho)

    # Lq: Average number of customers in the queue (excluding the one being served)
    Lq = (rho ** 2) / (1 - rho)

    # W: Average time a customer spends in the system (waiting + service)
    W = 1 / (service_rate - arrival_rate)

    # Wq: Average time a customer spends waiting in the queue
    Wq = rho / (service_rate - arrival_rate)

    # P0: Probability that there are 0 customers in the system (idle state)
    P0 = 1 - rho

    return {
        'utilization': round(rho, 4),  # Server utilization
        'L': round(L, 4),              # Avg # in system
        'Lq': round(Lq, 4),            # Avg # in queue
        'W': round(W, 4),              # Avg time in system
        'Wq': round(Wq, 4),            # Avg time in queue
        'P0': round(P0, 4)             # Idle probability
    }
