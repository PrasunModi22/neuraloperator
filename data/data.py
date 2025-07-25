import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

# ===== Three-body simulation =====
def generate_simulation(num_steps=500, dt=0.01, G=1.0, min_distance=0.1):
    """
    Generates a single 3-body simulation trajectory.
    Includes collision avoidance (repulsion) when bodies get too close.

    Returns:
        np.ndarray: A trajectory of shape [num_steps, 12], where 12
                    represents (pos_x, pos_y, vel_x, vel_y) for 3 bodies.
    """
    # Initial random positions and velocities
    pos = np.random.randn(3, 2) * 2 - 1 # 3 bodies in 2D, more spread out
    vel = np.random.randn(3, 2) * 0.1   # Small random velocities
    masses = np.random.uniform(0.5, 1.5, size=3) # Varying masses

    trajectory = [] # Stores flattened state [p1x,p1y,v1x,v1y, p2x,p2y,v2x,v2y, p3x,p3y,v3x,v3y]

    for _ in range(num_steps):
        # Store the current flattened state
        current_state = np.concatenate([pos.flatten(), vel.flatten()])
        trajectory.append(current_state)

        acc = np.zeros((3, 2))
        for i in range(3):
            for j in range(3):
                if i != j:
                    r_vec = pos[j] - pos[i]
                    dist = np.linalg.norm(r_vec)
                    
                    # Apply strong repulsion if too close to prevent division by zero and extreme forces
                    if dist < min_distance:
                        direction = (pos[i] - pos[j]) # Repel away from each other
                        # Scale repulsion inversely to distance squared, with a large constant
                        acc[i] += direction / (dist**2 + 1e-5) * G * 100 
                    else:
                        # Gravitational force
                        acc[i] += G * masses[j] * r_vec / (dist**3 + 1e-5) # Add small epsilon for stability

        vel += acc * dt
        pos += vel * dt

    return np.array(trajectory).astype(np.float32)

# ===== Save multiple simulations =====
def generate_and_save_data(num_sims=1000, num_steps=500, filename="three_body_data.pt"):
    """
    Generates multiple 3-body simulations and saves them to a PyTorch tensor file.

    Args:
        num_sims (int): Number of simulations to generate.
        num_steps (int): Number of time steps per simulation.
        filename (str): Name of the file to save the data.
    """
    print(f"Generating {num_sims} simulations, each with {num_steps} steps...")
    all_simulations = []
    for i in range(num_sims):
        sim_data = generate_simulation(num_steps=num_steps)
        all_simulations.append(sim_data)
        if (i + 1) % 100 == 0:
            print(f"Generated {i+1}/{num_sims} simulations.")

    # Stack all simulations: [num_sims, num_steps, 12]
    tensor_data = torch.tensor(np.stack(all_simulations), dtype=torch.float32)
    torch.save(tensor_data, filename)
    print(f"All simulations saved to {filename} with shape: {tensor_data.shape}")

# ===== Dataset class =====
class ThreeBodyDataset(Dataset):
    """
    PyTorch Dataset for 3-body simulation data.
    Uses the first simulation slice and provides input with a history of frames.
    Normalizes the entire dataset upon initialization.
    """
    def __init__(self, filename="three_body_data.pt", history_frames=2):
        """
        Args:
            filename (str): Path to the saved simulation data file.
            history_frames (int): Number of past frames (including current) to use as input.
                                  e.g., 2 means input is [t-1, t], target is [t+1].
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Data file not found at {filename}. Please run data.py to generate it.")

        self.history_frames = history_frames
        # Load all data, shape: [num_sims, T, 12]
        self.raw_data = torch.load(filename) 

        # Use only the first simulation as requested, shape: [T, 12]
        self.single_simulation = self.raw_data[0] 

        self.inputs = []
        self.targets = []

        # Prepare (input_sequence, target_frame) pairs
        # Input sequence: frames from (t - history_frames + 1) to t
        # Target: frame at t + 1
        # Loop starts from `history_frames - 1` to ensure enough preceding frames for input
        # Loop ends at `len(self.single_simulation) - 1` because the target is `t + 1`
        for t in range(self.history_frames - 1, len(self.single_simulation) - 1):
            input_sequence = self.single_simulation[t - self.history_frames + 1 : t + 1]
            self.inputs.append(input_sequence.reshape(-1)) # Flatten to [history_frames * 12]

            target_frame = self.single_simulation[t + 1]
            self.targets.append(target_frame.reshape(-1)) # Flatten to [12]

        self.inputs = torch.stack(self.inputs)   # Shape: [num_samples, history_frames * 12]
        self.targets = torch.stack(self.targets) # Shape: [num_samples, 12]

        print(f"Dataset created from first simulation. Input shape: {self.inputs.shape}, Target shape: {self.targets.shape}")

        # Normalize the entire dataset during initialization
        self._normalize_data()

    def _normalize_data(self):
        """Calculates and applies normalization (mean/std) to the entire dataset."""
        # Calculate mean and std across all samples and features
        self.input_mean = self.inputs.mean(dim=0, keepdim=True)
        self.input_std = self.inputs.std(dim=0, keepdim=True) + 1e-8 # Add epsilon for stability
        
        self.target_mean = self.targets.mean(dim=0, keepdim=True)
        self.target_std = self.targets.std(dim=0, keepdim=True) + 1e-8

        self.inputs = (self.inputs - self.input_mean) / self.input_std
        self.targets = (self.targets - self.target_mean) / self.target_std
        print("Dataset normalized.")

    def unnormalize_output(self, normalized_output: torch.Tensor) -> torch.Tensor:
        """Unnormalizes a model's output using the target's mean and std."""
        return normalized_output * self.target_std + self.target_mean

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# ===== Run this script directly to generate data =====
if __name__ == "__main__":
    # Generate 1000 simulations, each with 500 steps
    generate_and_save_data(num_sims=1000, num_steps=500)
    
    # Test the dataset loading and structure
    try:
        # history_frames=2 means input will be 2 * 12 = 24 features
        dataset = ThreeBodyDataset(history_frames=2) 
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        print("\nTesting dataset batches:")
        for i, (x, y) in enumerate(loader):
            print(f"Batch {i}: Input shape={x.shape}, Target shape={y.shape}")
            if i > 2: # Print a few batches to verify
                break
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred during dataset test: {e}")

