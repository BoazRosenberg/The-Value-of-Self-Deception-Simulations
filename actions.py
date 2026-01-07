import numpy as np
from functions import S_hat
from scipy.stats import norm
from value_functions import vfs


def evaluate_actions(vf, actions, state, beliefs, evaluation_resolution=100):
    """
    Compute expected subjective value of each action under each bias,
    then select the best action per bias deterministically.
    """

    # Discretized approximation to expectation over uncertainty
    Z_scores = np.linspace(-3, 3, evaluation_resolution)
    weights = norm.pdf(Z_scores)
    weights /= weights.sum()

    state_mu, state_sigma = state
    actions = np.asarray(actions)      # (n_actions, 2)
    beliefs = np.asarray(beliefs)      # (n_biases,)

    action_mus, action_sigmas = actions[:, 0], actions[:, 1]

    # Combine state, action, and bias additively in the mean
    mus = action_mus[:, None] + state_mu + beliefs[None, :]
    sigmas = np.sqrt(action_sigmas[:, None]**2 + state_sigma**2)

    # Sample outcome values and apply value function
    values = Z_scores[:, None, None] * sigmas[None, :, :] + mus[None, :, :]
    f_values = vf(values)

    # Expected subjective value for each (action, bias)
    evaluations = np.tensordot(weights, f_values, axes=([0], [0]))

    # Deterministic choice: one best action per bias
    choice_matrix = np.zeros_like(evaluations, dtype=int)
    best = np.argmax(evaluations, axis=0)
    choice_matrix[best, np.arange(len(beliefs))] = 1

    return evaluations, choice_matrix


def create_actions(beliefs, vf, state=(0, 1), evaluation_resolution=100,
                   mu_step=1, sigma_step=5, risk_seeking=False):
    """
    Construct a sequence of actions such that each new action is preferred
    only at a higher bias level than the previous ones.
    """

    actions = [(0.0, 0.0)]  # neutral baseline

    for i in range(1, len(beliefs)):
        current_biases = beliefs[:i + 1]
        prev_actions = np.asarray(actions)

        # Shift mean depending on risk attitude
        mu_dir = -1 if risk_seeking else 1
        new_mu = actions[-1][0] + mu_dir * mu_step

        low_sigma, high_sigma = actions[-1][1], None
        new_action = [new_mu, low_sigma + sigma_step]
        iterations = 0

        # Adjust sigma until the action is selected by exactly one bias
        while True:
            iterations += 1

            _, choices = evaluate_actions(
                vf=vf,
                actions=np.vstack([prev_actions, new_action]),
                state=state,
                beliefs=current_biases,
                evaluation_resolution=evaluation_resolution
            )

            selection_count = np.sum(choices, axis=1)
            if selection_count[-1] == 1:
                break

            too_good = selection_count[-1] > 1
            too_bad = selection_count[-1] == 0

            if risk_seeking:
                if too_good:
                    high_sigma = new_action[1]
                    new_action[1] = (low_sigma + high_sigma) / 2
                elif too_bad:
                    low_sigma = new_action[1]
                    new_action[1] += sigma_step if high_sigma is None else (low_sigma + high_sigma) / 2
            else:
                if too_good:
                    if high_sigma is None:
                        low_sigma = new_action[1]
                        new_action[1] += sigma_step
                    else:
                        low_sigma = new_action[1]
                        new_action[1] = (low_sigma + high_sigma) / 2
                elif too_bad:
                    high_sigma = new_action[1]
                    new_action[1] = (low_sigma + high_sigma) / 2

            if iterations > 100:
                raise RuntimeError(
                    f"Action search failed at bias index {i}. "
                    f"Last action: {new_action}. Actions so far: {actions}"
                )

        actions.append(new_action)

    return actions


if __name__ == "__main__": # Demonstrate an action generation and selection process

    biases = np.linspace(0, 1, 10)
    beliefs = S_hat(S=0, sigma=10, tau=biases, Q_means=True).flatten() # Map bias parameters to belief shifts

    vf_set = "diminishing"
    vf = vfs[vf_set][0]

    actions = create_actions(beliefs,
                             vf=vf,
                             risk_seeking=(vf_set == "increasing"))

    def plot_choices(actions, vf, beliefs=beliefs):
        """
        Show relative action values per bias (z-scored within bias),
        with stars indicating the chosen action.
        """

        import matplotlib.pyplot as plt

        evaluations, choices = evaluate_actions(
            vf=vf, actions=actions, state=(0, 1), beliefs=beliefs
        )

        # Standardize values within each bias for visualization
        evals_norm = (
            evaluations - evaluations.mean(axis=0, keepdims=True)
        ) / evaluations.std(axis=0, keepdims=True)

        plt.figure(figsize=(8, 6))
        plt.imshow(
            evals_norm, aspect='auto', cmap='Greys',
            extent=[biases.min(), biases.max(), len(actions), 0]
        )

        plt.colorbar(label='Standardized value (within bias)')
        plt.xlabel('Bias')
        plt.ylabel('Action index')
        plt.title('Action values and choices by bias')

        # Center stars on tiles
        x_step = (biases.max() - biases.min()) / len(biases)
        for b in range(len(biases)):
            a = np.argmax(choices[:, b])
            plt.plot(
                biases.min() + (b + 0.5) * x_step,
                a + 0.5,
                marker='*', color='red', markersize=12
            )

        plt.show()

    plot_choices(actions, vf=vf)
