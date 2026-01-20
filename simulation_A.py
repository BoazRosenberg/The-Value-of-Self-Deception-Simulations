from value_functions import vfs
import numpy as np
from define_simulations import run_simulation_A

if __name__ == "__main__":

    # value functions for each set are defined in value_functions.py
    # and could be adjusted by the user if desired

    for value_function_set_name in ["diminishing", "increasing"]:

        value_function_set = vfs[value_function_set_name]  # Choose the set of value functions to use

        n_copies = 30
        true_state = (0, 1)
        biases = np.linspace(0, 1, 9)
        n_actions = 9 # Number of actions to consider in the simulation
        observation_noise = 10

        df = run_simulation_A(
            value_function_set=value_function_set,
            true_state=true_state,
            observation_noise=observation_noise,
            biases=biases,
            n_copies=n_copies,
            name = value_function_set_name,
            risk_seeking= value_function_set_name == "increasing",
            Q_means = True
        )

        # save in csv_files folder
        df.to_csv(f"results/simulation_A_{value_function_set_name}.csv", index=False)
        print(f"Simulation complete. Results saved for value function set: {value_function_set_name}")



