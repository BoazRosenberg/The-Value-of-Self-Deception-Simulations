from value_functions import vfs
from define_simulations import run_simulation_B

if __name__ == "__main__":

    resolution = 200

    value_function_set_name = "diminishing"
    value_function_set = vfs[value_function_set_name]

    df = run_simulation_B(
                     value_function_set= value_function_set,
                     resolution=resolution,
                     bias_range=(0,1),
                     observation_noise_range=(0.1,15),
                     n_actions=9,
                     n_copies=50,
                     true_state=(0,1),
                     risk_seeking= value_function_set_name == "increasing" )

    df.to_csv(f"csv files/simulation_B.csv", index=False)
    print(f"Simulation complete. Results saved for agent: simulation_B")




