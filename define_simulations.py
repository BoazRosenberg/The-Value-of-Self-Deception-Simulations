from functions import *
from value_functions import *
from agent import Agent
from actions import create_actions
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import time



def run_simulation_A(value_function_set, risk_seeking = False, name = "", Q_means = False,

                     true_state=(0, 1), observation_noise=10, n_copies = 10,

                     biases=np.linspace(0.1, 0.9, 100), n_actions=10):
    """
    Run the agent simulation with specified parameters.

    This function sets up and runs a simulation of an agent with bias learning,
    using either diminishing or increasing value functions.
    """


    vf = value_function_set
    observation_noise = np.array([observation_noise])
    temp_biases = np.array([0.5])

    for i in tqdm(range(n_copies), desc=f"Running Simulation A: {name}", unit="iteration"):
        # Create actions for the agent
        actions = create_actions(
            beliefs= S_hat(S=true_state[0],
                                   sigma=observation_noise,
                                   tau= np.linspace(np.min(biases), np.max(biases), n_actions),
                                   Q_means = True).flatten(),
            vf=vf[0], # vf used for action selector
            state=true_state,
            risk_seeking= risk_seeking
        )



        # Initialize the agent
        agent = Agent(
            prior=true_state,
            change_cost=0.0,
            biases=           np.repeat(np.repeat(biases, len(temp_biases)), len(observation_noise)),
            temp_biases=      np.repeat(np.tile(temp_biases, len(biases)), len(observation_noise)),
            observation_noise=np.tile(observation_noise, len(biases) * len(temp_biases)),
            n_copies= 1,
            vf1=vf[0], # Action Selector
            vf2=vf[1], # Attentional Bias Controller
            actions= actions,
            name="A_" + name,
            Q_means= Q_means
        )

        # Run the simulation
        agent.single_epoch()
        temp_df = agent.df.copy()
        temp_df['copy'] = i  # Add a column to identify the copy number
        if i == 0:
            df = temp_df
        else:
            df = pd.concat([df, temp_df.dropna(axis=1, how='all')], ignore_index=True)

    return df


def run_simulation_B(value_function_set , true_state=(0, 1),resolution=10,
                     bias_range=(0,1), observation_noise_range=(0.1,50),
                     n_actions=10, n_copies=5, risk_seeking = False):

    biases = np.linspace(bias_range[0], bias_range[1], resolution)
    observation_noise = np.linspace(observation_noise_range[0], observation_noise_range[1], resolution)

    # Select value functions based on the chosen set
    vf = value_function_set

    for i in tqdm(range(n_copies), desc="Running Simulation B", unit="iteration"):

        # Create actions for the agent
        actions = create_actions(
            beliefs= S_hat(S=true_state[0],
                                   sigma=np.max(observation_noise),
                                   tau= np.linspace(np.min(biases), np.max(biases), n_actions)).flatten(),
            vf=vf[0], # action selector
            state=true_state,
            risk_seeking= risk_seeking
        )

        # Initialize the agent
        agent = Agent(
            prior=true_state,
            change_cost=0.0,
            biases=           np.repeat(biases, len(observation_noise)),
            temp_biases=      np.repeat(np.tile(0.5, len(biases)), len(observation_noise)),
            observation_noise=np.tile(observation_noise, len(biases) ),
            n_copies=1,
            vf1=vf[0], # Action Selector
            vf2=vf[1], # Attentional Bias Controller
            actions=actions,
            name="B"
        )

        # Run the simulation
        agent.single_epoch()
        temp_df = agent.df.copy()
        temp_df['copy'] = i  # Add a column to identify the copy number
        if i == 0:
            df = temp_df
        else:
            df = pd.concat([df, temp_df.dropna(axis=1, how='all')], ignore_index=True)

    print("Summarizing results...")
    # summarize by bias and observation_noise
    df_mean = df.groupby(['bias', 'observation_noise'],as_index=False).agg({
        'EV': 'mean',
        'S': 'mean',
        'action': 'mean'
    }).reset_index()

    return df_mean




def _run_single_copy_C(i, value_function_set, bias_range, temp_bias_range,
                       true_state, observation_noise, resolution,
                       n_actions, risk_seeking, Q_means,
                       change_cost, n_epochs, progress_counter):
    def report_progress():
        progress_counter.value += 1

    vf = value_function_set

    biases = np.linspace(bias_range[0], bias_range[1], resolution)
    temp_biases = np.linspace(temp_bias_range[0], temp_bias_range[1], resolution)

    additive_bias = S_hat(
        S=true_state[0],
        sigma=observation_noise * 2,
        tau=np.linspace(bias_range[0], bias_range[1], n_actions),
        Q_means=True
    )

    actions = create_actions(
        beliefs=additive_bias.flatten(),
        vf=vf[0],
        state=true_state,
        risk_seeking=risk_seeking
    )

    agent = Agent(
        prior=(0, 1),
        change_cost=change_cost,
        biases=np.repeat(np.repeat(biases, len(temp_biases)), len(observation_noise)),
        temp_biases=np.repeat(np.tile(temp_biases, len(biases)), len(observation_noise)),
        observation_noise=np.tile(observation_noise, len(biases) * len(temp_biases)),
        temp_noise=observation_noise,
        n_copies=1,
        vf1=vf[0],
        vf2=vf[1],
        actions=actions,
        Q_means=Q_means
    )

    agent.multi_epochs(n_epochs, progress_callback=report_progress)

    df = agent.df.copy()
    df["copy"] = i
    return df


def run_simulation_C(value_function_set, bias_range, temp_bias_range,
                     true_state, observation_noise, n_copies,
                     resolution=100, n_actions=10, risk_seeking=False,
                     Q_means=False, change_cost=0.1, n_epochs=40,
                     parallel=True, n_jobs=None):
    manager = Manager()
    progress_counter = manager.Value("i", 0)

    total_steps = n_copies * n_epochs
    dfs = []

    if parallel and n_copies > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(
                    _run_single_copy_C,
                    i,
                    value_function_set,
                    bias_range,
                    temp_bias_range,
                    true_state,
                    observation_noise,
                    resolution,
                    n_actions,
                    risk_seeking,
                    Q_means,
                    change_cost,
                    n_epochs,
                    progress_counter
                )
                for i in range(n_copies)
            ]

            pbar = tqdm(total=total_steps, desc="Running epochs", unit="epoch")

            last = 0
            while any(not f.done() for f in futures):
                current = progress_counter.value
                pbar.update(current - last)
                last = current
                time.sleep(0.1)

            for f in futures:
                dfs.append(f.result())

    else:
        for i in tqdm(range(n_copies), desc="Running copies", unit="copy"):
            dfs.append(
                _run_single_copy_C(
                    i,
                    value_function_set,
                    bias_range,
                    temp_bias_range,
                    true_state,
                    observation_noise,
                    resolution,
                    n_actions,
                    risk_seeking,
                    Q_means,
                    change_cost,
                    n_epochs,
                    progress_counter
                )
            )



    df = pd.concat(dfs, ignore_index=True)

    df_mean = (
        df.groupby(["bias", "temp_bias", "epoch"], as_index=False)
          .agg({
              "EV": "mean",
              "C": "mean",
              "action": "mean"
          })
          .reset_index(drop=True)
    )

    return df_mean
