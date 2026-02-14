import numpy as np
from copy import copy
import pandas as pd
from tqdm import tqdm
import os
import pickle
from scipy.stats import norm
from random import choices
import math
from numpy import argmax
import time
from functions import *



# Calculate the time left for the simulation based on the start time, total iterations, and current epoch
def time_left(start_time, epochs, current_epoch):
    current_time = time.time()
    time_passed = current_time - start_time
    time_left = time_passed * (epochs - current_epoch - 1)
    hours = int(time_left // 3600)
    minutes = int((time_left % 3600) // 60)
    seconds = int(time_left % 60)
    message =  f"Processing - Time left: {hours}h {minutes}m {seconds}s - Estimated End: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time + time_left))}"
    return message

# Define the agent and its components
class Agent:
    def __init__(self, biases, temp_biases, observation_noise, n_copies,
                 prior,
                 vf1, vf2, actions,temp_noise = 0, name="agent",
                 change_cost=0.1, forgetting_factor=0.17,
                 Q_means = False):

        # region General parameters
        self.name = name
        self.init_time = time.time()

        self.vf1 = vf1  # value function for model-free
        self.vf2 = vf2  # value function for model-based
        self.change_cost = change_cost
        self.Q_means = Q_means # controls whether simulation uses randomly sampled S_hats or mean S_hats
        # endregion

        # region Biases, noise and ids
        self.n_biases =          len(biases)
        self.biases =            biases
        self.temp_biases =       temp_biases


        # Observation noise
        self.observation_noise = observation_noise  # Constant
        self.temp_noise =        temp_noise         # Temporary
        self.sd_era = 0                             # Whether the current time point has
                                                    # temporary uncertainty or not (1 or 0, initially 0)

        self.n_copies = n_copies                    # Number of copies of the agent
        self.n_agents = self.n_biases * n_copies    # Total number of agents
        self.ids = range(self.n_agents)             # Id for each agent


        # endregion

        # region Initialize state
        self.prior = prior
        self.sigma =  np.repeat(self.observation_noise[:,np.newaxis], self.n_copies, axis = 1) * (1- forgetting_factor)
        self.S = np.full((self.n_biases, self.n_copies), float(prior[0]))  # true states
        self.calc_S_hat_components()
        self.update_S_hat()
        self.C =  np.zeros((self.n_biases, self.n_copies)) # cumulative cost

        # endregion

        # region actions

        self.action_mus = np.array([action[0] for action in actions])     # mus of different actions
        self.action_sigmas = np.array([action[1] for action in actions])  # sigmas of different actions

        self.action = np.full((self.n_biases, self.n_copies), 0)  # initial action index
        self.actions = actions

        self.integrate_action_and_state()
        # endregion

        # region Logs
        self.value_log = []
        self.action_log = []
        self.df = pd.DataFrame(columns=["agent", "bias", "epoch", "S", "S_hat", "C",
                                        "sigma", "action", "EV"])

        # endregion

    # region - Main methods

    # Update agents S_hat

    def calc_S_hat_components(self):

        self.S_hat_I = S_hat(S=self.S,
                            sigma=np.repeat(self.observation_noise[:, np.newaxis], self.n_copies, axis=1),
                            tau=self.biases,
                            Q_means = self.Q_means)

        # additional S_hat based on temporary uncertainty
        self.S_hat_II = S_hat(S=0,
                              sigma=np.repeat(self.temp_noise * self.observation_noise[:, np.newaxis] ,
                                           self.n_copies, axis=1),
                              tau=self.temp_biases,
                              Q_means = self.Q_means)

    def update_S_hat(self):
        # sd_era is 0 or 1 corresponding to whether the current epoch's transiently noisy information source
        # is noisy or not
        self.S_hat = self.S_hat_I + self.S_hat_II * self.sd_era 

    # Calculate the mu and sigma for expected PDFs for current action
    def integrate_action_and_state(self):
        self.S_with_action = self.S + self.action_mus[self.action]
        self.sigma_with_action = np.sqrt(self.sigma ** 2 + self.action_sigmas[self.action] ** 2)

    # Evaluate expected utility for all actions based on the current perceived state
    def evaluate_actions(self, vf=1, resolution=100):
        # output dimension : [n_actions, n_biases, n_copies]

        # Assign value function
        f = self.vf1 if vf == 1 else self.vf2

        # Prepare Z scores and densities
        Z_scores = np.linspace(-3, 3, resolution)
        densities = norm.pdf(Z_scores)
        normalized_densities = densities / np.sum(densities)

        # Calculate the expected PDFs for each action
        mus = self.S_hat[None, :, :] + self.action_mus[:, None, None]  # [ offer, agent]
        sigmas = np.sqrt(self.sigma[None, :, :] ** 2 + self.action_sigmas[:, None, None] ** 2)  # [ offer, agent]

        # Calculate the expected values for each action
        values = Z_scores[:, None, None, None] * sigmas[None, :, :, :] +\
                 mus[None, :, :,:]  # sample values for each action

        f_values = f(values)  # Evaluate samples
        expectancy = np.sum(normalized_densities[:, None, None, None] * f_values, axis=0)  # weighted average of samples

        return expectancy

    # Run a single epoch of the agent's decision-making process
    def single_epoch(self, record=True):

        # region - Choose action based on perceived state

        # cost of change : 0 for current action, 'change cost' for rest
        ps = np.array(range(len(self.actions))) # an array of action indices

        cost = (self.action[None, :, :] != ps[:, None, None]).astype(int) *\
               self.change_cost * self.action_mus[ps[:, None, None]]
        # a matrix of costs for changing to each action for each agent
        # if the new action is different from the current - a cost is paid


        self.action = argmax(self.evaluate_actions() - cost, axis=0) # Choose best action with taking cost into account
        self.action_log.append(self.action)

        self.integrate_action_and_state()
        # endregion

        # region - Pay cost and update perceived state
        x_idx, y_idx = np.indices(self.action.shape)
        cost_payed = cost[self.action, x_idx, y_idx]


        # evaluate current action
        self.v2_outcomes = transformed_expectancy(self.S_with_action,
                                                  self.sigma_with_action,
                                                  self.vf2) - \
                           cost_payed

        # update C
        self.C += cost_payed

        self.action_log.append(self.action)
        self.value_log.append(copy(self.v2_outcomes))

        if record:
            df = pd.DataFrame({"agent": self.ids,
                               "bias": np.repeat(self.biases, self.n_copies),
                               "temp_bias": np.repeat(self.temp_biases, self.n_copies),
                               "observation_noise": np.repeat(self.observation_noise, self.n_copies),
                               "epoch": len(self.value_log),
                               "S": self.S.flatten(),
                               "S_hat": self.S_hat.flatten(),
                               "C": self.C.flatten(),
                               "sigma": self.sigma.flatten(),
                               "action": self.action_log[-1].flatten(),
                               "EV": self.v2_outcomes.flatten()})

            self.df = pd.concat([self.df.dropna(axis=1, how='all'), df.dropna(axis=1, how='all')], ignore_index=True)

    # Run multiple epochs with temporary bias (simulation C)
    def multi_epochs(self, n_epochs, record=True, progress_callback=None):

        # Base cycle of temporary noise SDs
        cycle = np.array([0, 1])

        # Repeat cycle enough times and then truncate to n_epochs
        temp_sd_eras = np.tile(cycle, int(np.ceil(n_epochs / len(cycle))))[:n_epochs]

        n_epochs = len(temp_sd_eras)

        for i in range(n_epochs):

            self.sd_era = temp_sd_eras[i]
            self.update_S_hat()
            self.single_epoch(record=record)
            progress_callback()


    # endregion

    # region - Save results

    def save_model(self, path="saved models/", model_name=False):
        if not model_name:
            model_name = self.name
        with open(path + model_name + ".pkl", 'wb') as file:
            pickle.dump(self, file)
        print("Model saved as: ", model_name)

    def save_csv(self, path="results/", name=False):
        if not name:
            name = self.name
        self.df.to_csv(path + str(name) + "_results.csv")

    def update_csv(self, path="results/", name=False, agent_id=False):
        if not name:
            name = self.name

        if agent_id:  # add suffix to "agent" column
            self.df['agent'] = self.df['agent'].astype(str) + "_" + str(agent_id)

        file_path = path + str(name) + ".csv"

        if os.path.isfile(file_path):
            # File exists, append without header
            self.df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            # File does not exist, write with header
            self.df.to_csv(file_path, mode='w', header=True, index=False)

    # endregion