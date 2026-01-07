from functions import *
from value_functions import *
from agent import Agent
from actions import create_policies, evaluate_policies
import numpy as np
from value_functions import vfs
from define_simulations import run_simulation_C


if __name__ == "__main__":

    # region  - run simulation
    change_cost = 0.05
    n_epochs = 40

    n_copies = 100
    resolution = 200
    bias_range = (0,1)  # Range of biases to use in the simulation
    temp_bias_range = bias_range

    value_function_set =  ["diminishing","increasing"][0]  # Choose the set of value functions to use
    Q_means  = False
    vf = vfs[value_function_set]
    true_state = (0, 1)
    observation_noise = np.array([5])
    n_policies = 5

    df = run_simulation_C( value_function_set= vf,
                            bias_range= bias_range,
                            temp_bias_range= temp_bias_range,
                            true_state= true_state,
                            observation_noise= observation_noise,
                            n_copies= n_copies,
                            resolution= resolution,
                            n_policies= n_policies,
                            risk_seeking= value_function_set == "increasing",
                            change_cost= change_cost,
                           Q_means=  Q_means,
                            )

    # save in csv_files folder
    means = "_means" if Q_means else ""
    name = f"simulation_C_{value_function_set}"  + means
    df.to_csv(f"csv files/{name}.csv", index=False)

    print(f"Simulation complete. Results saved for agent: simulation_C_{value_function_set}")
