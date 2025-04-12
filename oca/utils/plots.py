"""
oca/utils/plots.py \n
Plotting utilities
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(log_path):
    episode_steps = []
    with open(log_path, "r") as file:
        for line in file:
            match = re.search(
                r"ep (\d+) done\. total_steps=(\d+) \| reward=([-\d.]+) \| episode_steps=(\d+)",
                line,
            )
            if match:
                ep = int(match.group(1))
                ep_steps = int(match.group(4))
                episode_steps.append((ep, ep_steps))
    return episode_steps


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    return np.convolve(y, box, mode="same")


def plot_steps_vs_episodes(log_dir, run_name, smooth_window=10, save_path=None):
    log_path = os.path.join(log_dir, run_name, "logger.log")
    data = parse_log_file(log_path)

    if not data:
        print("No data found in log.")
        return

    episodes, ep_steps = zip(*data)
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, ep_steps, alpha=0.2)
    plt.plot(episodes, smooth(ep_steps, smooth_window))
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.grid(True, linestyle=":")

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir",
        type=str,
        default="../runs/",
        help="Directory containing the run folders",
    )
    parser.add_argument(
        "--run_name", type=str, required=True, help="Name of the run folder"
    )
    parser.add_argument(
        "--smooth_window", type=int, default=10, help="Window size for smoothing"
    )
    parser.add_argument(
        "--save_path", type=str, default=None, help="Optional path to save the plot"
    )

    args = parser.parse_args()
    plot_steps_vs_episodes(
        args.log_dir, args.run_name, args.smooth_window, args.save_path
    )
