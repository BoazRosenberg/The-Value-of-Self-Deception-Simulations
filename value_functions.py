"""
Value functions for simulations.

Each function defines a transformation of outcome value `x` to reflect different assumptions
about marginal value—either decreasing or increasing. These functions can be used to simulate
different agent preferences or evaluative styles.
"""
import numpy as np

# regionSet 1: Decreasing marginal value

def MF(x):
    """Model-free system with diminishing returns."""
    return x - (np.power(np.exp(0.1),-x))

def MB(x):
    """Model-based system with diminishing returns."""
    return x - (np.power(np.exp(0.01),-x))

def MB_plus_MF(x):
    """Combined model-based and model-free systems with diminishing returns."""
    return (MB(x) + MF(x))/2
# endregion

# region Set 2: Increasing marginal value

def MF2(x):
    return x + (np.power(np.exp(0.1),x))

def MB2(x):
    return x + (np.power(np.exp(0.01),x))

def MB2_plus_MF2(x):
    return (MB2(x) + MF2(x))/2
# endregion

# region Define sets

# [V_1: Action Selector, V_2: Attentional Bias Controller]

vfs = {"diminishing": [MB_plus_MF,   MB],
       "increasing":  [MB2_plus_MF2, MB2], }

# endregion

# Test plots
if __name__ == "__main__":

    # Choose a set of value functions to plot
    chosen_set = vfs["increasing"]

    # The range of state values typically used in the simulation
    # We recommend for the value functions to deviate in this range
    # While avoiding sharp curvature changes

    x_range = np.linspace(-50, 50, 500)


    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for vf in chosen_set:
        plt.plot(x_range, vf(x_range), label=vf.__name__)
    plt.title("Value Functions")
    plt.xlabel("Outcome Value (x)")
    plt.ylabel("Transformed Value")
    plt.legend()
    plt.grid()
    plt.show()
