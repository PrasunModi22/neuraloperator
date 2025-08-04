import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import sys
from matplotlib.lines import Line2D

def generate_simulation(num_steps=500, dt=0.01, G=1.0, min_distance=0.1):
    """
    Generates a single 3-body simulation trajectory using a more accurate
    velocity Verlet integration scheme.
    """
    pos = np.random.randn(3, 2) * 2 - 1
    vel = np.random.randn(3, 2) * 0.1
    masses = np.random.uniform(0.5, 1.5, size=3)

    trajectory = []

    # Calculate initial acceleration
    acc = np.zeros((3, 2))
    for i in range(3):
        for j in range(3):
            if i != j:
                r_vec = pos[j] - pos[i]
                dist = np.linalg.norm(r_vec)
                if dist < min_distance:
                    direction = (pos[i] - pos[j])
                    acc[i] += direction / (dist**2 + 1e-5) * G * 100
                else:
                    acc[i] += G * masses[j] * r_vec / (dist**3 + 1e-5)
    acc /= masses[:, np.newaxis] 

    for _ in range(num_steps):
        # Save the current state before updating
        current_state = np.concatenate([pos.flatten(), vel.flatten()])
        trajectory.append(current_state)

        # Update velocity (half step)
        vel += acc * (dt / 2)

        # Update position
        pos += vel * dt

        # Calculate new acceleration
        new_acc = np.zeros((3, 2))
        for i in range(3):
            for j in range(3):
                if i != j:
                    r_vec = pos[j] - pos[i]
                    dist = np.linalg.norm(r_vec)
                    if dist < min_distance:
                        direction = (pos[i] - pos[j])
                        new_acc[i] += direction / (dist**2 + 1e-5) * G * 100
                    else:
                        new_acc[i] += G * masses[j] * r_vec / (dist**3 + 1e-5)
        new_acc /= masses[:, np.newaxis]

        # Update velocity (full step)
        vel += new_acc * (dt / 2)
        
        # Update acceleration for the next step
        acc = new_acc 

    return np.array(trajectory).astype(np.float32), masses.astype(np.float32)

def generate_and_save_data(num_sims=1000, num_steps=500, filename="three_body_data.pt"):
    """Generates multiple 3-body simulations and saves them to a PyTorch tensor file."""
    print(f"Generating {num_sims} simulations, each with {num_steps} steps...")
    all_trajectories = []
    all_masses = []
    for i in range(num_sims):
        sim_data, sim_masses = generate_simulation(num_steps=num_steps)
        all_trajectories.append(sim_data)
        all_masses.append(sim_masses)
        if (i + 1) % 100 == 0:
            print(f"Generated {i+1}/{num_sims} simulations.")

    tensor_data = torch.tensor(np.stack(all_trajectories), dtype=torch.float32)
    mass_data = torch.tensor(np.stack(all_masses), dtype=torch.float32)

    torch.save({'data': tensor_data, 'masses': mass_data}, filename)
    print(f"All simulations saved to {filename} with data shape: {tensor_data.shape}, masses shape: {mass_data.shape}")

class ThreeBodyDataset(Dataset):
    """
    PyTorch Dataset for 3-body simulation data.
    This version implements a single, global normalization across the entire dataset.
    """
    def __init__(self, filename="three_body_data.pt", history_frames=2, num_sims_to_use=10):
        if not os.path.exists(filename):
            print(f"Data file '{filename}' not found. Generating data...")
            generate_and_save_data(filename=filename)

        self.history_frames = history_frames
        
        loaded_content = torch.load(filename) 
        if not isinstance(loaded_content, dict) or 'data' not in loaded_content or 'masses' not in loaded_content:
            error_msg = (
                f"Error: Data file '{filename}' is not in the expected dictionary format. "
                "It should contain 'data' and 'masses' keys. "
            )
            raise ValueError(error_msg)

        # Use only the specified number of simulations
        self.raw_data = loaded_content['data'][:num_sims_to_use]
        self.raw_masses = loaded_content['masses'][:num_sims_to_use]
        print(f"Using the first {num_sims_to_use} simulations for training.")

        # --- CORRECTED GLOBAL NORMALIZATION ---
        self._normalize_data()

        self.inputs = []
        self.targets = []
        self.sample_masses = []

        # Create input/target pairs from the normalized data
        for sim_idx in range(len(self.normalized_data)):
            sim_data = self.normalized_data[sim_idx]
            sim_masses = self.raw_masses[sim_idx]
            for t in range(self.history_frames - 1, len(sim_data) - 1):
                input_sequence = sim_data[t - self.history_frames + 1 : t + 1]
                self.inputs.append(input_sequence.reshape(-1))

                target_frame = sim_data[t + 1]
                self.targets.append(target_frame.reshape(-1))
                
                self.sample_masses.append(sim_masses)

        self.inputs = torch.stack(self.inputs) 
        self.targets = torch.stack(self.targets) 
        self.sample_masses = torch.stack(self.sample_masses)

        print(f"Dataset created. Input shape: {self.inputs.shape}, Target shape: {self.targets.shape}")

    def _normalize_data(self):
        """Calculates and applies normalization (mean/std) to the entire dataset."""
        # Flatten all simulation data to a single tensor for consistent mean/std calculation
        flat_data = self.raw_data.view(-1, 12)
        
        # Calculate mean and std for each of the 12 features (pos and vel for 3 bodies)
        self.data_mean = flat_data.mean(dim=0, keepdim=True)
        self.data_std = flat_data.std(dim=0, keepdim=True) + 1e-8 # Add epsilon for stability
        
        # Apply normalization to all raw data
        self.normalized_data = (self.raw_data - self.data_mean) / self.data_std
        print("Entire dataset normalized with consistent mean/std.")

    def unnormalize(self, normalized_tensor: torch.Tensor) -> torch.Tensor:
        """Unnormalizes a tensor using the global mean and std."""
        # Ensure the tensor is the correct shape for unnormalization
        return normalized_tensor * self.data_std.to(normalized_tensor.device) + self.data_mean.to(normalized_tensor.device)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.sample_masses[idx]

# Visualization
def plot_3body_state(ax, state_vector_np, masses_np, title="", colors=['r', 'g', 'b'], 
                     vel_arrow_color='k', grav_arrow_color='purple', arrow_scale=0.5, alpha=0.8, G=1.0):
    """
    Plots the positions, velocities, and gravitational forces of 3 bodies.
    """
    states = state_vector_np.reshape(3, 4)
    positions = states[:, :2]
    velocities = states[:, 2:] 

    forces = np.zeros((3, 2))
    for i in range(3):
        for j in range(3):
            if i != j:
                r_vec = positions[j] - positions[i]
                dist = np.linalg.norm(r_vec)
                
                force_magnitude = (G * masses_np[j] * masses_np[i]) / (dist**2 + 1e-5)
                force_direction = r_vec / (dist + 1e-5)
                forces[i] += force_direction * force_magnitude

    for i in range(3):
        px, py = positions[i]
        vx, vy = velocities[i]
        fx, fy = forces[i]
        color = colors[i]
        
        ax.plot(px, py, 'o', color=color, markersize=8, alpha=alpha, label=f'Body {i+1}')
        
        ax.quiver(px, py, vx, vy, color=vel_arrow_color, scale=1, scale_units='xy', 
                  angles='xy', width=0.005, headwidth=3, headlength=5, alpha=alpha)
        
        force_norm = np.linalg.norm(forces[i])
        if force_norm > 1e-6: 
            fx_norm, fy_norm = fx / force_norm, fy / force_norm
        else:
            fx_norm, fy_norm = 0, 0
        
        ax.quiver(px, py, fx_norm, fy_norm, color=grav_arrow_color, scale=1, scale_units='xy', 
                  angles='xy', width=0.005, headwidth=3, headlength=5, alpha=alpha)
        
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([-3, 3]) 
    ax.set_ylim([-3, 3])
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)

# --- Main execution block for testing and visualization ---
if __name__ == "__main__":
    HISTORY_FRAMES = 2 
    DATA_FILENAME = "three_body_data.pt"
    VIS_PLOT_PATH = "dataset_visualization.png" 

    try:
        # The dataset will automatically generate data if it's missing
        dataset = ThreeBodyDataset(filename=DATA_FILENAME, history_frames=HISTORY_FRAMES, num_sims_to_use=1)
    except ValueError as e:
        print(e)
        sys.exit(1) 
    
    print("\nStarting dataset visualization...")
    # Get the first sample from the dataset
    x_sample_norm, y_sample_norm, masses_sample = dataset[0] 

    # The input `x_sample_norm` is a flattened tensor of `history_frames * 12` features.
    # The last 12 features correspond to the current state at time 't'.
    current_state_t_norm = x_sample_norm[-12:] 
    current_state_t_unnorm = dataset.unnormalize(current_state_t_norm).numpy()

    # The target `y_sample_norm` is the state at time 't+1'.
    gt_next_state_unnorm = dataset.unnormalize(y_sample_norm).numpy()

    fig_vis, axs_vis = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    fig_vis.suptitle("3-Body Dataset Visualization (Sample 0)", fontsize=16)

    plot_3body_state(axs_vis[0], current_state_t_unnorm, masses_sample.numpy(), 
                     title="Current State (Time t)", vel_arrow_color='k', grav_arrow_color='purple')

    plot_3body_state(axs_vis[1], gt_next_state_unnorm, masses_sample.numpy(), 
                     title="Ground Truth Next State (Time t+1)", vel_arrow_color='k', grav_arrow_color='purple')

    legend_elements = [
        Line2D([0], [0], marker='o', color='r', label='Body 1 Position', markersize=8),
        Line2D([0], [0], marker='o', color='g', label='Body 2 Position', markersize=8),
        Line2D([0], [0], marker='o', color='b', label='Body 3 Position', markersize=8),
        Line2D([0], [0], color='k', lw=2, label='Velocity Vector'),
        Line2D([0], [0], color='purple', lw=2, label='Grav. Force Vector')
    ]
    fig_vis.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 0.95))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.savefig(VIS_PLOT_PATH, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Dataset visualization plot saved to {VIS_PLOT_PATH}")
