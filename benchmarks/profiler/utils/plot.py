# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.interpolate import griddata

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def plot_prefill_performance(
    prefill_tp_size, prefill_ttft, prefill_thpt_per_gpu, target_ttft, output_dir
):
    """
    Plot prefill performance as a 2D scatter plot with TP size annotations.

    Args:
        prefill_tp_size: list of TP sizes
        prefill_ttft: list of time to first token values
        prefill_thpt_per_gpu: list of throughput per GPU values
        target_ttft: target TTFT value for the vertical line
        output_dir: directory to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(prefill_ttft, prefill_thpt_per_gpu, s=100)
    for i, tp in enumerate(prefill_tp_size):
        plt.annotate(
            f"TP{tp}",
            (prefill_ttft[i], prefill_thpt_per_gpu[i]),
            xytext=(10, 0),
            textcoords="offset points",
            fontsize=10,
        )

    plt.axvline(
        x=target_ttft, color="r", linestyle="--", label=f"Target TTFT: {target_ttft} ms"
    )
    plt.legend()

    plt.title("Prefill Performance")
    plt.xlabel("Time to First Token (ms)")
    plt.ylabel("Prefill throughput per GPU (tokens/s/GPU)")
    plt.grid(True)

    plot_path = f"{output_dir}/prefill_performance.png"
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Performance plot saved to {plot_path}")
    plt.close()


def plot_decode_performance(decode_results, target_itl, output_dir):
    """
    Plot decode performance with multiple TP size lines.

    Args:
        decode_results: list of tuples (tp_size, itl_list, thpt_per_gpu_list)
        target_itl: target ITL value for the vertical line
        output_dir: directory to save the plot
    """
    plt.figure(figsize=(10, 6))

    for tp_size, itl_list, thpt_per_gpu_list in decode_results:
        plt.plot(itl_list, thpt_per_gpu_list, label=f"TP{tp_size}")

    plt.axvline(
        x=target_itl, color="r", linestyle="--", label=f"Target ITL: {target_itl} ms"
    )
    plt.legend()
    plt.title("Decode Performance")
    plt.xlabel("Inter Token Latency (ms)")
    plt.ylabel("Decode throughput per GPU (tokens/s/GPU)")
    plt.grid(True)

    plot_path = f"{output_dir}/decode_performance.png"
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Performance plot saved to {plot_path}")
    plt.close()


def plot_prefill_interpolation(
    prefill_isl_np, prefill_ttft_np, prefill_thpt_per_gpu_np, work_dir
):
    """
    Plot TTFT and throughput vs ISL with quadratic interpolation.

    Args:
        prefill_isl_np: numpy array of input sequence lengths
        prefill_ttft_np: numpy array of time to first token values
        prefill_thpt_per_gpu_np: numpy array of throughput per GPU values
        work_dir: directory to save plots
    """
    # Fit quadratic functions
    ttft_coeffs = np.polyfit(prefill_isl_np, prefill_ttft_np, 2)
    thpt_coeffs = np.polyfit(prefill_isl_np, prefill_thpt_per_gpu_np, 2)

    # Create interpolation functions
    ttft_poly = np.poly1d(ttft_coeffs)
    thpt_poly = np.poly1d(thpt_coeffs)

    # Generate points for smooth curves
    x_interp = np.linspace(min(prefill_isl_np), max(prefill_isl_np), 100)
    ttft_interp = ttft_poly(x_interp)
    thpt_interp = thpt_poly(x_interp)

    # Plot TTFT vs ISL
    plt.figure(figsize=(10, 6))
    plt.scatter(prefill_isl_np, prefill_ttft_np, s=100, label="Measured data")
    plt.plot(
        x_interp,
        ttft_interp,
        "r-",
        label=f"Quadratic fit: {ttft_coeffs[0]:.2e}x² + {ttft_coeffs[1]:.2e}x + {ttft_coeffs[2]:.2e}",
    )

    plt.title("Prefill TTFT vs Input Sequence Length")
    plt.xlabel("Input Sequence Length (tokens)")
    plt.ylabel("Time to First Token (ms)")
    plt.grid(True)
    plt.legend()

    ttft_plot_path = f"{work_dir}/prefill_ttft_interpolation.png"
    plt.savefig(ttft_plot_path, dpi=300)
    logger.info(f"TTFT interpolation plot saved to {ttft_plot_path}")
    plt.close()

    # Plot Throughput vs ISL
    plt.figure(figsize=(10, 6))
    plt.scatter(prefill_isl_np, prefill_thpt_per_gpu_np, s=100, label="Measured data")
    plt.plot(
        x_interp,
        thpt_interp,
        "g-",
        label=f"Quadratic fit: {thpt_coeffs[0]:.2e}x² + {thpt_coeffs[1]:.2e}x + {thpt_coeffs[2]:.2e}",
    )

    plt.title("Prefill Throughput vs Input Sequence Length")
    plt.xlabel("Input Sequence Length (tokens)")
    plt.ylabel("Prefill throughput per GPU (tokens/s/GPU)")
    plt.grid(True)
    plt.legend()

    thpt_plot_path = f"{work_dir}/prefill_throughput_interpolation.png"
    plt.savefig(thpt_plot_path, dpi=300)
    logger.info(
        f"Prefill throughput per GPU interpolation plot saved to {thpt_plot_path}"
    )
    plt.close()


def plot_decode_3d_surface(x_kv_usage, y_context_length, z_itl, tp_size, work_dir):
    """
    Plot 3D surface for decode interpolation with KV usage, context length, and ITL.

    Args:
        x_kv_usage: list of KV usage percentages
        y_context_length: list of context lengths
        z_itl: list of ITL values
        tp_size: TP size for the plot filename
        work_dir: directory to save the plot
    """
    xi = np.linspace(min(x_kv_usage), max(x_kv_usage), 100)
    yi = np.linspace(min(y_context_length), max(y_context_length), 100)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((x_kv_usage, y_context_length), z_itl, (X, Y), method="cubic")

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")  # type: ignore

    # Create the surface plot with customizations
    surf = ax.plot_surface(  # type: ignore
        X,
        Y,
        Z,
        cmap=cm.coolwarm,  # type: ignore
        linewidth=0.2,
        antialiased=True,
        alpha=0.8,
    )

    # Add a color bar with custom settings
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label("Z Value", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Add labels with custom font sizes
    ax.set_xlabel("Active KV Percentage", fontsize=12)
    ax.set_ylabel("Decode Context Length", fontsize=12)
    ax.set_zlabel("ITL", fontsize=12)  # type: ignore

    # Set viewing angle
    ax.view_init(elev=30, azim=45)  # type: ignore
    ax.grid(True)
    ax.tick_params(axis="both", which="major", labelsize=10)

    plot_path = f"{work_dir}/decode_tp{tp_size}.png"
    logger.info(f"Saving ITL surface plot to {plot_path}")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
