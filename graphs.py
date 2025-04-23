import pandas as pd
import matplotlib.pyplot as plt

# Load your result CSV
df = pd.read_csv("all_combinations_results.csv")

def plot_parameter_analysis(param_name):
    # 1. Best-per-value (min of avg_cost per value)
    best_per_value = df.loc[df.groupby(param_name)["avg_cost"].idxmin()]

    # 2. Mean avg_cost per value
    mean_per_value = df.groupby(param_name)["avg_cost"].mean().reset_index()

    # Plot both
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot best
    ax.plot(best_per_value[param_name], best_per_value["avg_cost"], marker='o', label="Best config (lowest avg_cost)", linestyle='--')

    # Plot mean
    ax.plot(mean_per_value[param_name], mean_per_value["avg_cost"], marker='s', label="Mean avg_cost (across all configs)")

    ax.set_title(f"{param_name} vs Avg Cost")
    ax.set_xlabel(param_name)
    ax.set_ylabel("Avg Cost")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    # Save the plot to graphs directory
    plt.savefig(f"graphs/{param_name}_vs_avg_cost.png")
    plt.close()

# Example usage
plot_parameter_analysis("cooling_function")
plot_parameter_analysis("move_operator")
plot_parameter_analysis("initial_temp")
plot_parameter_analysis("cooling_alpha")
plot_parameter_analysis("max_iterations")
plot_parameter_analysis("min_temp")

