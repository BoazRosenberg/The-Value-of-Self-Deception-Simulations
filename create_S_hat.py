import numpy as np
import pandas as pd

def update_rule(Q, outcomes, eta_0, tau):
    # Q shape: (n_eta, n_tau, n_copies)
    # outcomes shape: (n_copies,)
    # eta_0, tau are 1D arrays of learning parameters

    # compute prediction error
    pe = outcomes[None, None, :] - Q  # broadcast to all eta/tau

    # bias factor (vectorized)
    pos_mask = pe >= 0
    b = np.where(pos_mask, (1 - tau)[None, :, None] / tau[None, :, None],
                           tau[None, :, None] / (1 - tau)[None, :, None])
    a = (1-eta_0[:, None, None])/eta_0[:, None, None]  # broadcast to all tau/copies
    # learning rate
    lr = 1 / (1 + a* b)

    # update
    Q = Q + lr * pe
    return Q

def learning_process(eta, tau, n_rounds=100, n_copies=1000):
    eta = np.array(eta)
    tau = np.array(tau)

    n_eta = len(eta)
    n_tau = len(tau)

    # initialize Q
    Q = np.full((n_eta, n_tau, n_copies), 0)

    # generate outcomes: standard normal samples per round and copy
    outcomes = np.random.normal(0, 1, size=(n_rounds, n_copies))

    # iterate over rounds (cannot vectorize across rounds due to recursion)
    for t in range(n_rounds):
        print(t)
        Q = update_rule(Q, outcomes[t], eta, tau)

    # Reshape Q into long format: (eta, tau, copy)
    eta_grid, tau_grid, copy_grid = np.meshgrid(eta, tau, np.arange(n_copies), indexing="ij")
    Qs = pd.DataFrame({
        "eta": eta_grid.ravel(),
        "tau": tau_grid.ravel(),
        "copy": copy_grid.ravel(),
        "Q": Q.ravel()
    })
    return Qs

def sample_beta_mode_concentration(mode, concentration, size=1):
    alpha = mode * (concentration - 2) + 1
    beta = (1 - mode) * (concentration - 2) + 1

    return np.random.beta(alpha, beta, size=size)

def run_default():
    eta = np.random.beta(2, 10, size=100)
    tau = np.linspace(0, 1, 100)

    Qs = learning_process(eta, tau, n_rounds=300, n_copies=100)
    Qs.to_csv("S_hat.csv", index=False)    # save to csv



if __name__ == "__main__":

    eta = np.random.beta(2, 10, size=100)
    tau = np.linspace(0, 1, 100)

    Qs = learning_process(eta, tau, n_rounds=300, n_copies=100)
    Qs.to_csv("S_hat.csv", index=False)    # save to csv
    print("Simulation completed and results saved.")