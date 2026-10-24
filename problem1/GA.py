import  numpy as np
import  matplotlib.pyplot as plt
from    scipy.integrate import solve_ivp

EVENTS = np.array([
    [+1,  0,  0],   # birth
    [-1, +1,  0],   # infection
    [ 0, -1, +1],   # recovery
    [-1,  0,  0],   # S dies
    [ 0, -1,  0],   # I dies
    [ 0,  0, -1],   # R dies
])

def get_rates(state, params):

    S, I, R = state
    beta = params["beta"]
    gamma = params["gamma"]
    mu = params["mu"]
    N0 = params["N0"]
    N = S + I + R

    return np.array([
        mu * N0,                              # birth
        beta * S * I / N if N > 0 else 0.0,   # infection
        gamma * I,                            # recovery
        mu * S,                               # S dies
        mu * I,                               # I dies
        mu * R,                               # R dies
    ])

def first_reaction_step(rates, rng):

    taus = np.full(len(rates), np.inf)
    mask = rates > 0
    taus[mask] = rng.exponential(1.0 / rates[mask])
    event = np.argmin(taus)

    return taus[event], event

def gillespie(state0, params, t_max, seed=None):

    rng = np.random.default_rng(seed)
    t, state = 0.0, np.array(state0, dtype=int)
    times, states = [t], [state.copy()]

    while t < t_max:
        rates = get_rates(state, params)
        if rates.sum() == 0:
            break
        tau, event = first_reaction_step(rates, rng)
        t += tau
        state = state + EVENTS[event]
        times.append(t)
        states.append(state.copy())

    return np.array(times), np.array(states)

def ode_rhs(t, state, params):

    rates = get_rates(state, params)

    return np.dot(EVENTS.T, rates)


def deterministic(state0, params, t_max, n_points=1000):

    t_eval = np.linspace(0, t_max, n_points)
    sol = solve_ivp(
        ode_rhs, (0, t_max), state0, args=(params,), t_eval=t_eval
    )

    return sol.t, sol.y.T


def resample(times, states, t_grid):

    idx = np.searchsorted(times, t_grid, side="right") - 1

    return states[idx]


def run_ensemble(state0, params, t_max, n_runs, t_grid):

    runs = np.empty((n_runs, len(t_grid), 3))
    for k in range(n_runs):
        times, states = gillespie(state0, params, t_max, seed=k)
        runs[k] = resample(times, states, t_grid)

    return runs

def covariance(runs, i, j):

    a = runs[:, :, i]
    b = runs[:, :, j]
    n_runs = runs.shape[0]
    deviations = (a - a.mean(axis=0)) * (b - b.mean(axis=0))

    return deviations.sum(axis=0) / (n_runs - 1)