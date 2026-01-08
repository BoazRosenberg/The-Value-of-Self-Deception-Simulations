import numpy as np
from scipy.stats import norm
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import math
from scipy.integrate import quad
import scipy.stats as stats
from scipy.optimize import minimize_scalar
import pandas as pd
import os


# Approximate the perceived mu for different biases and noise
# Samples from a df created in "estimate terminal Q.py"

def load_S_hat_df():
    dir = "estimate learning processes"
    df = pd.read_csv(os.path.join(dir, "S_hat.csv"))[['tau', 'Q']]
    mean_Q = df.groupby(['tau'])['Q'].mean().reset_index()
    grouped = df.groupby('tau')['Q'].apply(np.array)
    taus = grouped.index.to_numpy()
    Q_arrays = grouped.to_numpy()
    mean_Q_array = mean_Q['Q'].to_numpy()
    return taus, Q_arrays, mean_Q_array

taus_grid, Q_by_tau, mean_Q_by_tau = load_S_hat_df()


def S_hat(S, sigma, tau, Q_means = False,
          taus_grid = taus_grid, Q_by_tau = Q_by_tau, mean_Q_by_tau = mean_Q_by_tau):
    # Approximated the agent's learnt belief about its state

    tau = np.atleast_1d(tau)
    Q_sampled = np.array([np.random.choice(qs) for qs in Q_by_tau]) if not Q_means else mean_Q_by_tau
    # interpolate sampled curve
    Q_out = np.interp(tau, taus_grid, Q_sampled)

    S_hat_out = Q_out[:, None] * sigma + S
    return S_hat_out


def transformed_expectancy(mu, sigma, f, resolution=100):
    Z_scores = np.linspace(-3, 3, resolution)
    densities = norm.pdf(Z_scores)
    normalized_densities = densities / np.sum(densities)

    values = Z_scores[:, None, None] * sigma[None, :, :] + mu[None, :, :]
    f_values = f(values)
    expectancy = np.sum(normalized_densities[:, None, None] * f_values, axis=0)

    return expectancy

def evaluate_actions_by_mu(actions, vf , sigma = 10, observation_noise = 30,
                            mu_range=[-10, 100], biases = [0.1,0.3,0.5,0.7,0.9],
                            resolution=100):

        # region - evaluate actions by mu
        # create z scores for weighted E[x] calculation
        Z_scores = np.linspace(-3, 3, resolution)
        densities = norm.pdf(Z_scores)
        normalized_densities = densities / np.sum(densities)

        # organize the true and perceived mus
        true_mus = np.linspace(mu_range[0], mu_range[1], resolution)
        p_mus = np.array([S_hat(mu, observation_noise, biases) for mu in true_mus])

        # calculate the expected PDFs for each action
        action_mus = np.array([action[0] for action in actions])
        action_sigmas = np.array([action[1] for action in actions])

        mus = p_mus + action_mus[None,None,:]  # [mus, bias, action]
        sigmas = np.sqrt(sigma ** 2 + action_sigmas ** 2)  # [action]

        # calculate the expected values for each action
        values = Z_scores[:, None, None,None] * sigmas[None, None,None, :] + mus[None, :, :,:]  # sample values for each action
        f_values = vf(values)  # evaluate samples    [Z_values, mus, bias, action]

        evaluations = np.sum(normalized_densities[:, None, None,None] * f_values, axis=0)  # weighted average of samples
        choice = np.argmax(evaluations, axis=2)  # [mus, bias]
        # endregion

        # region - plot results
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(true_mus, biases, choice.T, shading='auto', cmap='viridis')
        plt.colorbar(label='Optimal action')
        plt.xlabel('true mu')
        plt.ylabel('Bias')
        plt.title('Optimal action heatmap')
        plt.show()
        # endregion


def evaluate_actions_by_observation_noise(actions, vf, mu = 0,
                            observation_noise_range=[1,50], biases=[0.1, 0.3, 0.5, 0.7, 0.9],
                            resolution=100):
    # region - evaluate actions by mu
    # create z scores for weighted E[x] calculation
    Z_scores = np.linspace(-3, 3, resolution)
    densities = norm.pdf(Z_scores)
    normalized_densities = densities / np.sum(densities)

    # organize the true and perceived mus

    observation_noise = np.linspace(observation_noise_range[0], observation_noise_range[1], resolution)
    sigma = observation_noise * 0.8303150
    p_mus = np.array([S_hat(mu, noise, biases) for noise in observation_noise])

    # calculate the expected PDFs for each action
    action_mus = np.array([action[0] for action in actions])
    action_sigmas = np.array([action[1] for action in actions])

    mus = p_mus + action_mus[None, None, :]  # [observation_noise, bias, action]
    sigmas = np.sqrt(sigma[:,None,None] ** 2 + action_sigmas[None,None,:] ** 2)  # [observation_noise, bias action]

    # calculate the expected values for each action
    values = Z_scores[:   , None, None, None] *\
             sigmas  [None, :   , :   , :] + \
             mus     [None, :   , :   , :]  # sample values for each action
    f_values = vf(values)  # evaluate samples    [Z_values, mus, bias, action]

    evaluations = np.sum(normalized_densities[:, None, None, None] * f_values, axis=0)  # weighted average of samples
    choice = np.argmax(evaluations, axis=2)  # [mus, bias]
    # endregion

    # region - plot results
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(observation_noise, biases, choice.T, shading='auto', cmap='viridis')
    plt.colorbar(label='Optimal action')
    plt.xlabel('observation noise')
    plt.ylabel('Bias')
    plt.title('Optimal action heatmap')
    plt.show()
    # endregion




# region ## Agents ###

def load_model( model_name , path = "saved models//"):
    with open(path + model_name + ".pkl", 'rb') as file:
        model = pickle.load(file)
    return model


# endregion

# region ### math ####

def logit(x):
    x= np.array(x)
    return np.log(x/(1-x))

def sigmoid(x):
    x = np.array(x)
    return 1/(1+np.exp(-x))

def sum_sigmoidal(a,b):
    # sums two values between 0 and 1
    # to a value between 0 and 1
    #
    # logit transformation
    a,b = np.array(a), np.array(b)
    a_continous = np.log(a/(1-a))
    b_continous = np.log(b/(1-b))
    sum_continous = (a_continous + b_continous)
    # sigmoid back to 0-1
    sum_discrete = 1/(1+np.exp(-sum_continous))
    return sum_discrete


# endregion

def choose_action(vf, mu, sigma, actions, resolution=100, output = "choice"):
    # evaluate actions for each agent - according to its perceived current state
    # vf: function
    # mu, sigma = np.arrays for perceived states
    #actions = list of tuples (mu, sigma)

    # assign value function
    f = vf

    action_mus = np.array([action[0] for action in actions])
    action_sigmas = np.array([action[1] for action in actions])

    # prepare Z scores and densities
    Z_scores = np.linspace(-3, 3, resolution)
    densities = norm.pdf(Z_scores)
    normalized_densities = densities / np.sum(densities)

    # calculate the expected PDFs for each action
    mus = mu[np.newaxis, :] + action_mus[:, np.newaxis]  # [ offer, agent]
    sigmas = np.sqrt(sigma[np.newaxis, :] ** 2 + action_sigmas[:, np.newaxis] ** 2)  # [ offer, agent]

    # calculate the expected values for each action
    values = Z_scores[:, None, None] * sigmas[None, :, :] + mus[None, :, :]  # sample values for each action
    f_values = f(values)  # evaluate samples
    expectancy = np.sum(normalized_densities[:, None, None] * f_values, axis=0)  # weighted average of samples

    if output == "expectancy":
        return expectancy
    elif output == "choice":
        choice = np.argmax(expectancy, axis=0)
        return choice

def generate_offers(mu, sigma, n):
    # mu = (mu, sigma), sd - (mu,sigma)
    # for the normal distribution of mu and sigma

    offers = [(float(np.random.normal(mu[0], mu[1], 1)),
               float(np.random.normal(sigma[0], sigma[1], 1))) for _ in range(n)]
    return offers

def t_ci(data, confidence=0.95):
    a = np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return m - h, m + h

