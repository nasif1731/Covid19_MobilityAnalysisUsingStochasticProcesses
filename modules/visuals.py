import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Flask/production use

import matplotlib.pyplot as plt
import os

def plot_steady_pie(steady, labels, out_path):
    """
    Plots a pie chart showing steady-state probabilities of each mobility state.

    Args:
        steady (list of float): Steady-state probabilities.
        labels (list of str): Corresponding labels for states.
        out_path (str): Path to save the pie chart image.
    """
    plt.figure(figsize=(5, 5))
    plt.pie(steady, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Long-Term Mobility Distribution")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def plot_state_timeline(sequence, save_path, csv_path=None):
    """
    Plots a timeline of observed mobility states and optionally exports as CSV.

    Args:
        sequence (list of str): Sequence of states ('Low', 'Moderate', 'High').
        save_path (str): Path to save timeline plot image.
        csv_path (str): Optional path to save CSV of timeline.
    """
    import csv

    state_map = {"Low": 0, "Moderate": 1, "High": 2}
    y_labels = ["Low", "Moderate", "High"]
    mapped_states = [state_map.get(s, -1) for s in sequence]
    x = list(range(len(mapped_states)))

    # Downsample if too long
    MAX_POINTS = 500
    if len(x) > MAX_POINTS:
        step = len(x) // MAX_POINTS
        x = x[::step]
        mapped_states = mapped_states[::step]

    # Save to CSV if requested
    if csv_path:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Day", "State"])
            reverse_map = {v: k for k, v in state_map.items()}
            for i in range(len(x)):
                writer.writerow([x[i], reverse_map.get(mapped_states[i], 'Unknown')])

    # Plotting
    plt.figure(figsize=(12, 4))
    plt.plot(x, mapped_states, color='royalblue', linewidth=1.5, alpha=0.85)

    # Vertical lines at state changes
    for i in range(1, len(mapped_states)):
        if mapped_states[i] != mapped_states[i - 1]:
            plt.axvline(x[i], color='gray', linestyle=':', alpha=0.2)

    plt.yticks([0, 1, 2], y_labels)
    plt.xlabel("Day")
    plt.ylabel("Mobility State")
    plt.title("📈 Mobility State Timeline")
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_viterbi_path(states, out_path):
    """
    Plots the most likely hidden states using Viterbi algorithm.

    Args:
        states (list of str): Viterbi-decoded sequence of hidden states.
        out_path (str): Path to save the line chart.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(states)), states, drawstyle='steps-post', marker='o', color='mediumseagreen')
    plt.title("Most Likely Hidden States (Viterbi Path)")
    plt.xlabel("Day")
    plt.ylabel("Hidden State")
    plt.xticks(range(len(states)))
    plt.yticks(sorted(set(states)))
    plt.gca().set_yticklabels(sorted(set(states)), rotation=20)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_hidden_steady_pie(steady_dict, out_path):
    """
    Plots a pie chart of steady-state probabilities from hidden states.

    Args:
        steady_dict (dict): State → probability mapping.
        out_path (str): Path to save chart image.
    """
    labels = list(steady_dict.keys())
    sizes = list(steady_dict.values())

    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Steady-State Distribution (Hidden States)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_mm1_summary(metrics, out_path):
    """
    Plots bar chart for key M/M/1 Queueing metrics.

    Args:
        metrics (dict): Dictionary with keys 'L', 'Lq', 'W', 'Wq'.
        out_path (str): Path to save image.
    """
    labels = ['Avg Customers (L)', 'Avg in Queue (Lq)', 'Avg Time in System (W)', 'Avg Wait Time (Wq)']
    values = [metrics['L'], metrics['Lq'], metrics['W'], metrics['Wq']]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, values, color='teal')
    plt.title('M/M/1 Queueing Summary')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

