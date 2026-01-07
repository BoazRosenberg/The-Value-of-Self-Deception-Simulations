from value_functions import vfs
import numpy as np
from define_simulations import run_simulation_A

if __name__ == "__main__":

    value_function_set_name = ["diminishing", "increasing"][0]
    value_function_set = vfs[value_function_set_name]  # Choose the set of value functions to use
    name = value_function_set_name

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
        name = name,
        risk_seeking= value_function_set_name == "increasing",
        Q_means = True
    )

    # save in csv_files folder
    df.to_csv(f"csv files/simulation_A_{name}.csv", index=False)
    print(f"Simulation complete. Results saved for agent: {name}")

    # plot results based on df (Learnt Value by bias)

    import matplotlib.pyplot as plt
    df_mean = df.groupby('bias')['EV'].mean().reset_index()
    plt.figure(figsize=(8, 6))
    plt.plot(df_mean['bias'], df_mean['EV'], marker='o')
    plt.title(f"Learnt Value by Bias")
    plt.xlabel("Bias")
    plt.ylabel("Learnt Value")
    plt.grid()
    plt.show()



