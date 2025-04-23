import pandas as pd

# Load the results CSV
df = pd.read_csv("./all_combinations_results.csv")

# 1. Best configurations overall
best_config = df.nsmallest(5, "avg_cost")

# 2. Most stable configurations (lowest std deviation)
most_stable = df.nsmallest(5, "std_cost")

# 3. Fastest decent solutions (avg_cost < 8000)
fast_decent = df[df["avg_cost"] < 8000].nsmallest(5, "avg_time_sec")

# 4. Worst configurations
worst_config = df.nlargest(5, "avg_cost")

# 5. Average cost by cooling function
cooling_group = df.groupby("cooling_function")["avg_cost"].mean().sort_values()

# 6. Average cost by move operator
move_group = df.groupby("move_operator")["avg_cost"].mean().sort_values()

# 7. Performance at low initial temperature
low_init_temp = df[df["initial_temp"] < 500][["avg_cost", "std_cost"]].describe()

# 8. Performance when alpha = 0.999
alpha_999 = df[df["cooling_alpha"] == 0.999].groupby("cooling_function")["avg_cost"].mean()

# 9. Long run configurations (max_iterations > 10,000)
long_runs = df[df["max_iterations"] > 10000][["avg_cost", "std_cost", "avg_time_sec"]].describe()

# Print or save summaries
print("=== Best Configs ===\n", best_config)
print("\n=== Most Stable ===\n", most_stable)
print("\n=== Fastest Decent ===\n", fast_decent)
print("\n=== Worst Configs ===\n", worst_config)
print("\n=== Avg Cost by Cooling Function ===\n", cooling_group)
print("\n=== Avg Cost by Move Operator ===\n", move_group)
print("\n=== Low Initial Temp Stats ===\n", low_init_temp)
print("\n=== Alpha = 0.999 by Cooling Function ===\n", alpha_999)
print("\n=== Long Runs (>10k iterations) Stats ===\n", long_runs)
