import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor

# Ensure output folder exists
output_dir = "sa_experiments"
os.makedirs(output_dir, exist_ok=True)

# Extended parameter ranges (10 values each)
initial_temps = list(np.linspace(100, 4000, 8, dtype=int))
cooling_alphas = list(np.linspace(0.7, 0.999, 8))
max_iterations_list = list(np.linspace(250, 15000, 8, dtype=int))
min_temps = list(np.geomspace(1e-5, 0.3, 8))  # log scale

move_operators = ["two_opt", "swap", "insert", "three_opt"]

cooling_functions = {
    "linear_cooling": linear_cooling,
    "geometric_cooling": geometric_cooling,
    "logarithmic_cooling": logarithmic_cooling
}
# Prepare combinations
param_combinations = list(itertools.product(
    cooling_functions.items(), move_operators, initial_temps, cooling_alphas, max_iterations_list, min_temps
))

# Worker function
def run_sa_combo(combo):
    (cooling_name, cooling_func), move_op, init_temp, alpha, max_iter, min_temp = combo
    trial_costs = []
    trial_times = []

    for _ in range(3):
        start = time.time()
        _, cost, _, acc = simulated_annealing(
            cooling_function=lambda t, a, i: cooling_func(t, alpha, i),
            max_iterations=max_iter,
            initial_temp=init_temp,
            alpha=alpha,
            min_temp=min_temp,
            move_operator=move_op
        )
        end = time.time()
        trial_costs.append(cost)
        trial_times.append(end - start)

    return {
        "cooling_function": cooling_name,
        "move_operator": move_op,
        "initial_temp": init_temp,
        "cooling_alpha": alpha,
        "max_iterations": max_iter,
        "min_temp": min_temp,
        "avg_cost": sum(trial_costs) / len(trial_costs),
        "std_cost": pd.Series(trial_costs).std(),
        "avg_time_sec": sum(trial_times) / len(trial_times),
    }

# Run in parallel with 20 workers
print("Running full parameter grid with multiprocessing (20 workers)...")
with ProcessPoolExecutor(max_workers=40) as executor:
    results = list(tqdm(executor.map(run_sa_combo, param_combinations), total=len(param_combinations)))

# Save sorted results
df = pd.DataFrame(results)
df.sort_values(by="avg_cost", inplace=True)
csv_path = os.path.join(output_dir, "all_combinations_results.csv")
df.to_csv(csv_path, index=False)
print(f"Saved sorted results to: {csv_path}")

# Generate 6 best-value plots
plot_params = {
    "move_operator": list(set(move_operators)),
    "cooling_function": list(cooling_functions.keys()),
    "initial_temp": initial_temps,
    "cooling_alpha": cooling_alphas,
    "max_iterations": max_iterations_list,
    "min_temp": min_temps,
}

for param_name, param_values in plot_params.items():
    best_rows = []
    for val in param_values:
        subset = df[df[param_name] == val]
        if not subset.empty:
            best_row = subset.loc[subset["avg_cost"].idxmin()]
            best_rows.append((val, best_row["avg_cost"]))

    # Format labels to show only significant digits
    def format_label(x):
        if isinstance(x, float):
            return f"{x:.3g}" if x < 0.01 else f"{x:.2f}"
        return str(x)

    x_vals, y_vals = zip(*best_rows)
    x_labels = list(map(format_label, x_vals))

    plt.figure(figsize=(10, 5))
    plt.plot(x_labels, y_vals, marker='o')
    plt.title(f"{param_name} (best config per value) vs Avg Cost")
    plt.xlabel(param_name)
    plt.ylabel("Avg Cost")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{param_name}_best_vs_cost.png"))
    plt.close()
