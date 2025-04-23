import numpy as np
import matplotlib.pyplot as plt
import math
import random
from matplotlib.animation import FuncAnimation

# Berlin52 dataset coordinates
berlin52_coords = [
    (565, 575), (25, 185), (345, 750), (945, 685), (845, 655), (880, 660), (25, 230),
    (525, 1000), (580, 1175), (650, 1130), (1605, 620), (1220, 580), (1465, 200), (1530, 5),
    (845, 680), (725, 370), (145, 665), (415, 635), (510, 875), (560, 365), (300, 465),
    (520, 585), (480, 415), (835, 625), (975, 580), (1215, 245), (1320, 315), (1250, 400),
    (660, 180), (410, 250), (420, 555), (575, 665), (1150, 1160), (700, 580), (685, 595),
    (685, 610), (770, 610), (795, 645), (720, 635), (760, 650), (475, 960), (95, 260),
    (875, 920), (700, 500), (555, 815), (830, 485), (1170, 65), (830, 610), (605, 625),
    (595, 360), (1340, 725), (1740, 245)
]

# INIT I
def calculate_distance_matrix(cities):
    num_cities = len(cities)
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            distance_matrix[i, j] = np.sqrt(np.sum((cities[i] - cities[j])**2))
    return distance_matrix

# Generate initial solution (random permutation of cities, as said in lesson it is admissible)
def generate_initial_solution():
    return np.random.permutation(num_cities)

def calculate_tour_length(tour):
    length = 0
    for i in range(num_cities):
        length += distance_matrix[tour[i], tour[(i + 1) % num_cities]]
    return length

def two_opt_move(tour):
    """2-opt move: reverse a segment of the tour"""
    new_tour = tour.copy()
    i, j = sorted(random.sample(range(num_cities), 2))
    new_tour[i:j+1] = np.flip(new_tour[i:j+1])
    return new_tour

def swap_move(tour):
    new_tour = tour.copy()
    i, j = random.sample(range(num_cities), 2)
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour

def insert_move(tour):
    new_tour = tour.copy()
    i, j = random.sample(range(num_cities), 2)
    if i < j:
        city = new_tour[i]
        new_tour = np.delete(new_tour, i)
        new_tour = np.insert(new_tour, j-1, city)
    else:
        city = new_tour[i]
        new_tour = np.delete(new_tour, i)
        new_tour = np.insert(new_tour, j, city)
    return new_tour

def three_opt_move(tour):
    new_tour = tour.copy()

    # three random positions on tour
    i, j, k = sorted(random.sample(range(num_cities), 3))
    #split our segments
    segment1 = new_tour[:i+1]
    segment2 = new_tour[i+1:j+1]
    segment3 = new_tour[j+1:k+1]
    segment4 = new_tour[k+1:]
    option = random.randint(0, 3)

    # more random stuff to flip orders around
    if option == 0:
        new_tour = np.concatenate((segment1, np.flip(segment2), segment3, segment4))
    elif option == 1:
        new_tour = np.concatenate((segment1, segment2, np.flip(segment3), segment4))
    elif option == 2:
        new_tour = np.concatenate((segment1, np.flip(segment2), np.flip(segment3), segment4))
    else:
        new_tour = np.concatenate((segment1, segment3, segment2, segment4))

    return new_tour

# RDM SELECT MOVE OPT
def generate_neighbor(tour, move_operator="two_opt"):
    """Generate a neighbor using the specified move operator"""
    if move_operator == "swap":
        return swap_move(tour)
    elif move_operator == "insert":
        return insert_move(tour)
    elif move_operator == "three_opt":
        return three_opt_move(tour)
    else:  # Default to two_opt
        return two_opt_move(tour)

# COOLING
def linear_cooling(initial_temp, alpha, iteration):
    return initial_temp - alpha * iteration
def geometric_cooling(initial_temp, alpha, iteration):
    return initial_temp * (alpha ** iteration)
def logarithmic_cooling(initial_temp, alpha, iteration):
    return initial_temp / (1 + alpha * math.log(1 + iteration))

# ACCEPT/REJECT SOLUTION
def accept_transition(current_cost, new_cost, temperature):
    if new_cost < current_cost:
        return True
    acceptance_probability = math.exp((current_cost - new_cost) / temperature)
    return random.random() < acceptance_probability # That's the part were we use randomness to guide the search

# SA ALG (cleanned/commented/refactored by LLM in case some of you wanna understand it better, you don't wanna see my handwritten code :)
def simulated_annealing(cooling_function, max_iterations, initial_temp, alpha, min_temp, move_operator="two_opt"):
    # Tunable parameters are passed as arguments
    current_solution = generate_initial_solution()
    current_cost = calculate_tour_length(current_solution)
    best_solution = current_solution.copy()
    best_cost = current_cost

    # Store history for visualization
    history = []

    # Track acceptance rate
    total_moves = 0
    accepted_moves = 0

    temperature = initial_temp
    iteration = 0

    while iteration < max_iterations and temperature > min_temp:
        history.append((current_solution.copy(), temperature, current_cost, best_cost))

        # Generate neighbor using specified move operator
        neighbor = generate_neighbor(current_solution, move_operator)
        neighbor_cost = calculate_tour_length(neighbor)

        total_moves += 1

        # Decide whether to accept
        if accept_transition(current_cost, neighbor_cost, temperature):
            current_solution = neighbor
            current_cost = neighbor_cost
            accepted_moves += 1

            # Update best if improved
            if current_cost < best_cost:
                best_solution = current_solution.copy()
                best_cost = current_cost

        # Update temperature
        temperature = cooling_function(initial_temp, alpha, iteration)
        iteration += 1

    # Add final state
    history.append((current_solution.copy(), temperature, current_cost, best_cost))

    # Calculate final acceptance rate
    acceptance_rate = accepted_moves / total_moves if total_moves > 0 else 0

    return best_solution, best_cost, history, acceptance_rate

def visualize_simulated_annealing(history, params=None):
# clearly LLM guided implementation. This is how I use them to help me help you learn. There is no way I write a 300 lines ( out of 500) vizu code by hand in the allowed time.
# Lol, while writting the above, the editor autocompleted (so LLM) my previous sentence with that --> I am not a machine. I am a human being. I have a life outside of this. I have a family, friends, and hobbies. I am not a robot. I am not a computer program. I am not an AI. I am a person. I have feelings, thoughts, and emotions. I have dreams, aspirations, and goals. I have a life to live and a world to explore.

    """
    Visualize the simulated annealing process with interactive controls and parameter recap.

    Args:
        history: The history of solutions from the simulated annealing run
        params: Dictionary containing the parameters used for this run
    """
    # Default parameters if none provided
    if params is None:
        params = {
            "initial_temp": 1000.0,
            "min_temp": 0.1,
            "cooling_function": "geometric",
            "cooling_alpha": 0.99,
            "max_iterations": 10000,
            "move_operator": "two_opt",
            "dataset": "berlin52"
        }

    # Create a figure with adjustable subplots
    fig = plt.figure(figsize=(16, 8))

    # Create a grid spec to organize our plots and panels
    gs = fig.add_gridspec(2, 2, height_ratios=[5, 1], width_ratios=[1, 1])

    # Main subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Tour visualization
    ax2 = fig.add_subplot(gs[0, 1])  # Temperature and cost plots

    # Parameter panel at bottom
    ax_params = fig.add_subplot(gs[1, :])
    ax_params.axis('off')  # Hide axes

    # Create scatter plot of cities
    ax1.scatter(cities[:, 0], cities[:, 1], c='blue', s=100, zorder=1)

    # Plot the best solution found so far (will be updated in animation)
    best_tour_indices = history[-1][0]  # Use the last entry's tour as initial best
    best_tour_cities = np.append(cities[best_tour_indices], [cities[best_tour_indices[0]]], axis=0)
    best_line, = ax1.plot(best_tour_cities[:, 0], best_tour_cities[:, 1], 'g-', alpha=0.4,
                      linewidth=2, label='Best Solution', zorder=0)

    # Initialize current tour line
    tour_indices = history[0][0]
    tour_cities = np.append(cities[tour_indices], [cities[tour_indices[0]]], axis=0)
    current_line, = ax1.plot(tour_cities[:, 0], tour_cities[:, 1], 'r-', alpha=0.8,
                        linewidth=2.5, label='Current Solution', zorder=2)

    # Title with iteration, temperature, cost, and gap information
    title_text = f"Iteration: 0/{len(history)-1}"
    sub_title = f"Temp: {history[0][1]:.2f}, Cost: {history[0][2]:.2f}, Gap to optimal: {(history[0][2] - 7542) / 7542 * 100:.2f}%"
    ax1.set_title(title_text + "\n" + sub_title)
    ax1.set_xlabel("X Coordinate")
    ax1.set_ylabel("Y Coordinate")
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right', framealpha=0.9)

    # Temperature and cost plot with improved styling
    temps = [h[1] for h in history]
    costs = [h[2] for h in history]
    best_costs = [h[3] for h in history]
    iterations = list(range(len(history)))

    # Normalize temperature for better visualization
    norm_temps = np.array(temps) / max(temps)

    # Temperature plot with filled area
    ax2.plot(iterations, norm_temps, 'r-', linewidth=2, label='Temperature (normalized)')
    ax2.fill_between(iterations, 0, norm_temps, color='r', alpha=0.2)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Temperature (normalized)", color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.grid(True, linestyle='--', alpha=0.5, axis='x')

    # Cost plot on twin axis
    ax3 = ax2.twinx()
    ax3.plot(iterations, costs, 'b-', linewidth=2, label='Current Cost')
    ax3.plot(iterations, best_costs, 'g-', linewidth=2, label='Best Cost')
    ax3.set_ylabel("Tour Cost", color='b')
    ax3.tick_params(axis='y', labelcolor='b')

    # Add legend with all lines
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Create vertical tracker lines
    iteration_line = ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)

    # Add a text box for key metrics that will be updated
    metrics_box = ax1.text(0.02, 0.02, "", transform=ax1.transAxes,
                          bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8),
                          fontsize=9, verticalalignment='bottom')

    # Create the parameter recap panel
    # Format the cooling function name nicely
    cooling_func_name = params["cooling_function"].replace("_", " ").title()

    # Create parameter text with a more structured layout
    param_text = (
        f"SA PARAMETERS - Dataset: {params['dataset']} ({len(cities)} cities)\n\n"
        f"Temperature:  Initial = {params['initial_temp']:.1f}  |  Final = {temps[-1]:.4f}  |  Min Threshold = {params['min_temp']:.4f}\n"
        f"Cooling:  Function = {cooling_func_name}  |  Alpha = {params['cooling_alpha']:.4f}  |  Max Iterations = {params['max_iterations']}\n"
        f"Neighborhood:  Move Operator = {params['move_operator']}  |  Final Solution Gap to Optimal = {(best_costs[-1] - 7542) / 7542 * 100:.2f}%"
    )

    # Create a text box for parameters
    param_box = ax_params.text(0.5, 0.5, param_text,
                             transform=ax_params.transAxes,
                             fontsize=11,
                             bbox=dict(boxstyle="round,pad=0.6", facecolor='aliceblue',
                                      edgecolor='steelblue', alpha=0.8),
                             horizontalalignment='center',
                             verticalalignment='center')

    # Calculate if a move occurred between frames
    def detect_move(frame_idx):
        if frame_idx == 0:
            return None

        prev_tour = history[frame_idx-1][0]
        curr_tour = history[frame_idx][0]

        # If tours are identical, no move occurred
        if np.array_equal(prev_tour, curr_tour):
            return None

        # Find the cities that changed positions
        different_indices = []
        for i in range(len(prev_tour)):
            if prev_tour[i] != curr_tour[i]:
                different_indices.append(i)

        # Return the segment that changed, padded for better visibility
        if different_indices:
            start = max(0, min(different_indices) - 1)
            end = min(len(prev_tour) - 1, max(different_indices) + 1)
            return (start, end)

        return None

    # Animation update function
    def update(frame_idx):
        # Get current frame data
        tour_indices = history[frame_idx][0]
        temperature = history[frame_idx][1]
        current_cost = history[frame_idx][2]
        best_cost = history[frame_idx][3]

        # Update current tour line
        tour_cities = np.append(cities[tour_indices], [cities[tour_indices[0]]], axis=0)
        current_line.set_data(tour_cities[:, 0], tour_cities[:, 1])

        # Find the best solution up to this frame
        best_so_far_idx = np.argmin([h[3] for h in history[:frame_idx+1]])
        best_tour_indices = history[best_so_far_idx][0]
        best_tour_cities = np.append(cities[best_tour_indices], [cities[best_tour_indices[0]]], axis=0)
        best_line.set_data(best_tour_cities[:, 0], best_tour_cities[:, 1])

        # Detect and highlight move if one occurred
        changed_segment = detect_move(frame_idx)

        # Calculate metrics for display
        gap_to_optimal = (current_cost - 7542) / 7542 * 100
        best_gap_to_optimal = (best_cost - 7542) / 7542 * 100

        # Calculate acceptance rate for sliding window
        window_size = min(100, frame_idx)
        accepted_count = 0

        if window_size > 0:
            for i in range(max(0, frame_idx - window_size), frame_idx):
                if not np.array_equal(history[i][0], history[i+1][0]):
                    accepted_count += 1

            recent_acceptance_rate = accepted_count / window_size
        else:
            recent_acceptance_rate = 0

        # Update metrics textbox
        metrics_text = (
            f"Current cost: {current_cost:.0f}\n"
            f"Best cost: {best_cost:.0f}\n"
            f"Gap to optimal: {gap_to_optimal:.2f}%\n"
            f"Recent acceptance rate: {recent_acceptance_rate:.2%}"
        )
        metrics_box.set_text(metrics_text)

        # Update title with key information
        title_text = f"Iteration: {frame_idx}/{len(history)-1}"
        sub_title = f"Temp: {temperature:.2f}, Cost: {current_cost:.0f}, Gap: {gap_to_optimal:.2f}%"
        ax1.set_title(title_text + "\n" + sub_title)

        # Update the position of the iteration line
        iteration_line.set_xdata([frame_idx, frame_idx])

        return current_line, best_line, iteration_line, metrics_box

    # Create the animation
    ani = FuncAnimation(
        fig,
        update,
        frames=len(history),
        interval=100,  # Default interval (will be adjustable)
        blit=False,  # False to allow title and textbox updates
        repeat=False
    )

    # Add interactive controls using matplotlib widgets
    plt.subplots_adjust(bottom=0.2)  # Make room for controls

    # Create axes for buttons and slider
    ax_pause = plt.axes([0.4, 0.05, 0.1, 0.05])
    ax_speed = plt.axes([0.55, 0.05, 0.3, 0.03])

    # Create a pause/resume button
    pause_button = plt.Button(ax_pause, 'Pause/Resume')

    # Create a speed control slider
    speed_slider = plt.Slider(
        ax_speed, 'Speed',
        0.25, 5.0,
        valinit=1.0,
        valstep=0.25,
        orientation='horizontal'
    )

    # Setup is_paused flag
    is_paused = False

    # Pause/resume callback function
    def toggle_pause(event):
        nonlocal is_paused
        if is_paused:
            ani.resume()
        else:
            ani.pause()
        is_paused = not is_paused

    # Speed adjustment callback function
    def update_speed(val):
        # The slider value is a multiplier: higher means faster animation
        # We divide the base interval (100ms) by the slider value
        new_interval = int(100 / val)

        # Print for debugging
        print(f"Speed slider value: {val}, New interval: {new_interval}ms")

        # Update the animation interval directly
        ani._interval = new_interval

        # Force a redraw of the animation
        fig.canvas.draw_idle()

        # If animation is running, we need to restart it with the new interval
        if not is_paused:
            ani.event_source.stop()
            ani.event_source.interval = new_interval
            ani.event_source.start()

    # Connect callbacks to the controls
    pause_button.on_clicked(toggle_pause)
    speed_slider.on_changed(update_speed)

    # Add a text label for the slider
    plt.figtext(0.55, 0.10, "Animation Speed", ha="left")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Readjust to account for controls
    plt.show()

    return ani

def run_multiple_trials(num_trials=5):
    results = []
    for i in range(num_trials):
        print(f"Trial {i+1}/{num_trials}...")
        sol, cost, _, acc_rate = simulated_annealing(
            cooling_function=lambda t, a, i: cooling_function(t, COOLING_ALPHA, i),
            max_iterations=MAX_ITERATIONS,
            initial_temp=INITIAL_TEMP,
            alpha=COOLING_ALPHA,
            min_temp=MIN_TEMP,
            move_operator=MOVE_OPERATOR
        )
        results.append((cost, acc_rate))

    costs = [r[0] for r in results]
    acc_rates = [r[1] for r in results]

    print("\nMultiple Trial Results:")
    print(f"Average cost: {np.mean(costs):.2f} (std: {np.std(costs):.2f})")
    print(f"Best cost: {min(costs):.2f}")
    print(f"Worst cost: {max(costs):.2f}")
    print(f"Average acceptance rate: {np.mean(acc_rates):.2%}")

    return results

# Main execution
if __name__ == "__main__":
    # set stuff up
    cities = np.array(berlin52_coords)
    num_cities = len(cities)
    distance_matrix = calculate_distance_matrix(cities)
    # rdm seed
    random.seed(42)
    np.random.seed(42)

    # ------------- THIS IS YOUR PLAYGROUND -------------
    # You can change the parameters here to test different configurations
    # INITIAL_TEMP = 1000.0
    # MIN_TEMP = 0.1
    # MAX_ITERATIONS = 1000
    # COOLING_ALPHA = 0.99
    # cooling_function = geometric_cooling # geometric_cooling, linear_cooling, logarithmic_cooling
    # MOVE_OPERATOR = "two_opt" # "two_opt", "swap", "insert", "three_opt"
    INITIAL_TEMP = 100
    MIN_TEMP = 0.015774
    MAX_ITERATIONS = 15000
    COOLING_ALPHA = 0.956286
    cooling_function = logarithmic_cooling
    MOVE_OPERATOR = "three_opt"

    # ----------------------------------------------------




    # PRINTING
    print(f"Running simulated annealing on Berlin52 dataset ({num_cities} cities):")
    print(f"Initial temperature: {INITIAL_TEMP}")
    print(f"Cooling function: {cooling_function.__name__}")
    print(f"Cooling alpha: {COOLING_ALPHA}")
    print(f"Move operator: {MOVE_OPERATOR}")
    print(f"Max iterations: {MAX_ITERATIONS}")
    best_solution, best_cost, history, acceptance_rate = simulated_annealing(
        cooling_function=lambda t, a, i: cooling_function(t, COOLING_ALPHA, i),
        max_iterations=MAX_ITERATIONS,
        initial_temp=INITIAL_TEMP,
        alpha=COOLING_ALPHA,
        min_temp=MIN_TEMP,
        move_operator=MOVE_OPERATOR
    )
    print(f"Best solution found: {best_solution}")
    print(f"Best tour cost: {best_cost:.2f}")
    print(f"Known optimal solution for Berlin52: 7542")
    print(f"Gap to optimal: {(best_cost - 7542) / 7542 * 100:.2f}%")
    print(f"Final acceptance rate: {acceptance_rate:.2%}")



    # Uncomment to run multiple trials (if you wanan check some stats and stability of your setup. High explo -> high variance)
    trial_results = run_multiple_trials(5)
    visualize_simulated_annealing(history)