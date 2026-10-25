import numpy as np
import pandas as pd
import networkx as nx
NET_COLORS = {"ER": "tab:blue", "WS": "tab:orange", "BA": "tab:green"}

def sir_step(G, state, beta, gamma, rng):
    new_state = state.copy()

    for node in G.nodes():
        if state[node] == 0:
            k = 0
            for nb in G.neighbors(node):
                if state[nb] == 1:
                    k += 1
            if k > 0 and rng.random() < 1 - (1 - beta) ** k:
                new_state[node] = 1

        elif state[node] == 1:
            if rng.random() < gamma:
                new_state[node] = 2

    return new_state


def run_sir(G, beta, gamma, initial_infected, n_iter=300, seed=None):
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()

    state = np.zeros(n, dtype=int)
    state[initial_infected] = 1

    S, I, R = [], [], []
    for t in range(n_iter):
        S.append(np.sum(state == 0) / n)
        I.append(np.sum(state == 1) / n)
        R.append(np.sum(state == 2) / n)

        if np.sum(state == 1) == 0:
            break

        state = sir_step(G, state, beta, gamma, rng)

    return np.array(S), np.array(I), np.array(R)


def run_many(G, beta, gamma, n_init, n_runs, seed=0):
    n = G.number_of_nodes()
    results = []

    for r in range(n_runs):
        rng = np.random.default_rng(seed + r)
        initial = rng.choice(n, size=n_init, replace=False)
        S, I, R = run_sir(G, beta, gamma, initial, seed=seed + r)
        results.append((S, I, R))

    return results

def make_network(kind, n, k, seed, p_rewire=0.1):
    if kind == "ER":
        return nx.erdos_renyi_graph(n, k / (n - 1), seed=seed)
    if kind == "WS":
        return nx.watts_strogatz_graph(n, k, p_rewire, seed=seed)
    if kind == "BA":
        return nx.barabasi_albert_graph(n, k // 2, seed=seed)


def network_stats(G):
    degrees = np.array([d for node, d in G.degree()])
    giant = G.subgraph(max(nx.connected_components(G), key=len))

    return {
        "avg degree": np.mean(degrees),
        "max degree": np.max(degrees),
        "<k^2>/<k>": np.mean(degrees ** 2) / np.mean(degrees),
        "clustering": nx.average_clustering(G),
        "avg path length": nx.average_shortest_path_length(giant),
        "max betweenness": max(nx.betweenness_centrality(G).values()),
        "max closeness": max(nx.closeness_centrality(G).values()),
    }
    
def load_socio(path="transmission_network.csv"):
    adj = pd.read_csv(path, sep=";", index_col=0)
    A = (adj.to_numpy() > 0).astype(int)
    return nx.from_numpy_array(A)

def choose_tests(A, degrees, vaccinated, recent_pos, last_tested, t, n_tests, rng):
    n = len(degrees)
    score = 1000 * (A @ recent_pos) + degrees + rng.random(n) * 0.1
    ok = ~vaccinated & (recent_pos == 0) & (t - last_tested > 3)
    candidates = np.where(ok)[0]
    order = candidates[np.argsort(-score[candidates])]
    return order[:n_tests]


def choose_vaccinations(strategy, degrees, A, vaccinated, recent_pos, recent_neg, n_vacc, rng, accuracy=1.0):
    n = len(degrees)
    if strategy == "none" or n_vacc == 0:
        return []

    if strategy == "random":
        candidates = np.where(~vaccinated)[0]
        return rng.choice(candidates, size=min(n_vacc, len(candidates)), replace=False)

    if strategy == "degree":
        candidates = np.where(~vaccinated)[0]
        score = degrees + rng.random(n) * 0.1

    if strategy == "combined":
        candidates = np.where(~vaccinated & (recent_pos == 0))[0]
        score = 1000 * (A @ recent_pos) + 100 * recent_neg + degrees + rng.random(n) * 0.1

    if strategy == "adaptive":
        trust = 2 * accuracy - 1
        candidates = np.where(~vaccinated & ((recent_pos == 0) | (trust == 0)))[0]
        score = trust * (50 * (A @ recent_pos) + 10 * recent_neg) + degrees + rng.random(n) * 0.1

    order = candidates[np.argsort(-score[candidates])]
    return order[:n_vacc]


def run_campaign(G, beta, gamma, strategy, n_vacc, n_tests, accuracy, seed,
                 n_init=5, n_iter=300, memory=5):
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    degrees = np.array([d for node, d in G.degree()])
    A = nx.to_numpy_array(G)

    state = np.zeros(n, dtype=int)
    initial = np.random.default_rng(seed).choice(n, size=n_init, replace=False)
    state[initial] = 1
    ever_infected = state == 1

    vaccinated = np.zeros(n, dtype=bool)
    last_pos = np.full(n, -1000)
    last_neg = np.full(n, -1000)
    last_tested = np.full(n, -1000)
    wasted = 0
    I = []

    for t in range(n_iter):
        I.append(np.sum(state == 1) / n)
        if np.sum(state == 1) == 0:
            break

        if strategy in ["combined", "adaptive"] and n_tests > 0:
            recent_pos = (t - last_pos <= memory).astype(float)
            tested = choose_tests(A, degrees, vaccinated, recent_pos, last_tested, t, n_tests, rng)
            for node in tested:
                infected = state[node] == 1
                if rng.random() < accuracy:
                    result = infected
                else:
                    result = not infected
                if result:
                    last_pos[node] = t
                else:
                    last_neg[node] = t
                last_tested[node] = t

        recent_pos = (t - last_pos <= memory).astype(float)
        recent_neg = ((t - last_neg <= memory) & (recent_pos == 0)).astype(float)
        targets = choose_vaccinations(strategy, degrees, A, vaccinated, recent_pos, recent_neg, n_vacc, rng, accuracy)
        
        
        for node in targets:
            vaccinated[node] = True
            if state[node] == 0:
                state[node] = 2
            else:
                wasted += 1

        state = sir_step(G, state, beta, gamma, rng)
        ever_infected = ever_infected | (state == 1)

    return {"final_size": np.mean(ever_infected), "peak": max(I), "wasted": wasted, "I": np.array(I)}

def run_sir_ndlib(G, beta, gamma, initial_infected, n_iter=300, seed=None):
    import ndlib.models.ModelConfig as mc
    import ndlib.models.epidemics as ep

    np.random.seed(seed)
    model = ep.SIRModel(G, seed=seed)

    config = mc.Configuration()
    config.add_model_parameter("beta", beta)
    config.add_model_parameter("gamma", gamma)
    config.add_model_initial_configuration("Infected", [int(v) for v in initial_infected])
    model.set_initial_status(config)

    iterations = model.iteration_bunch(n_iter, node_status=False, progress_bar=False)
    counts = model.build_trends(iterations)[0]["trends"]["node_count"]

    n = G.number_of_nodes()
    return np.array(counts[0]) / n, np.array(counts[1]) / n, np.array(counts[2]) / n