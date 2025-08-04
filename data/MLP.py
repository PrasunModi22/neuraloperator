import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import sys
from matplotlib.lines import Line2D

# Assuming the data.py file is in the same directory and contains the ThreeBodyDataset
# class.
try:
    from data import ThreeBodyDataset
except ImportError:
    print("Error: Could not import ThreeBodyDataset from data.py.")
    print("Please ensure data.py is in the same directory.")
    sys.exit(1)


# ==============================================================================
# --- MLP Model Definition ---
# ==============================================================================
class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) for predicting the next state
    of a 3-body system. This model flattens the history frames and processes
    them as a single vector.
    """
    def __init__(self, input_features, output_features, hidden_size=256):
        """
        Args:
            input_features (int): Total number of features in the input (e.g., history_frames * 12).
            output_features (int): Number of features in the output (e.g., 12).
            hidden_size (int): Size of the hidden layers.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_features)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_features)
        """
        original_shape = x.shape
        
        # Handle single sample case
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        x = self.net(x)
        
        # Return to original shape if input was a single sample
        if len(original_shape) == 1:
            x = x.squeeze(0)
            
        return x


# ==============================================================================
# --- Helper Functions for Evaluation and Plotting ---
# ==============================================================================

def calculate_center_of_mass(states, masses):
    """
    Calculates the center of mass for a set of three-body states.
    Args:
        states (torch.Tensor): Tensor of shape [num_steps, 12] containing state vectors.
        masses (torch.Tensor): Tensor of shape [3] with body masses.
    Returns:
        torch.Tensor: Tensor of shape [num_steps, 2] with the COM positions.
    """
    # Reshape states to [num_steps, 3 bodies, 4 features (x, y, vx, vy)]
    positions = states.view(-1, 3, 4)[..., :2]
    total_mass = masses.sum()
    # Expand masses to [1, 3, 1] for broadcasting
    weighted_positions = positions * masses.view(1, 3, 1)
    com_positions = weighted_positions.sum(dim=1) / total_mass
    return com_positions


def plot_trajectory(predicted_traj, ground_truth_traj, num_steps, plot_path, num_sims):
    """
    Plots the predicted and ground truth 2D trajectories of the three bodies.
    """
    plt.figure(figsize=(10, 8))
    
    # Reshape trajectories to [steps, bodies, features]
    pred_pos = predicted_traj.view(-1, 3, 4).cpu().numpy()
    gt_pos = ground_truth_traj.view(-1, 3, 4).cpu().numpy()
    
    colors = ['r', 'g', 'b']
    labels = ['Body 1', 'Body 2', 'Body 3']
    
    # Plot predicted trajectories
    for i in range(3):
        plt.plot(pred_pos[:, i, 0], pred_pos[:, i, 1], '--', color=colors[i], label=f'Predicted {labels[i]}')
        
    # Plot ground truth trajectories
    for i in range(3):
        plt.plot(gt_pos[:, i, 0], gt_pos[:, i, 1], '-', color=colors[i], label=f'Ground Truth {labels[i]}')
        
    plt.title(f"MLP Predicted vs. Ground Truth Trajectory ({num_sims} Sims, Steps 1 to {num_steps})")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Trajectory plot saved to {plot_path}")


def plot_com_trajectory(predicted_com, ground_truth_com, plot_path, num_sims):
    """
    Plots the center of mass trajectory for predictions and ground truth.
    """
    plt.figure(figsize=(10, 8))
    
    # Move tensors to CPU and convert to numpy for plotting
    predicted_com_np = predicted_com.cpu().numpy()
    ground_truth_com_np = ground_truth_com.cpu().numpy()

    plt.plot(predicted_com_np[:, 0], predicted_com_np[:, 1], 'ro-', label='Predicted COM')
    plt.plot(ground_truth_com_np[:, 0], ground_truth_com_np[:, 1], 'bo-', label='Ground Truth COM')
    
    plt.title(f"MLP Center of Mass Trajectory ({num_sims} Sims)")
    plt.xlabel("X Position of COM")
    plt.ylabel("Y Position of COM")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Center of mass plot saved to {plot_path}")


# ==============================================================================
# --- Training Configuration and Main Execution ---
# ==============================================================================

# --- Training Configuration ---
HISTORY_FRAMES = 2 
BATCH_SIZE = 64
NUM_EPOCHS = 500 
LEARNING_RATE = 1e-3
NUM_PREDICTION_STEPS = 10
# Change this variable to switch between 10, 100, and 1000 simulations
NUM_SIMULATIONS_TO_USE = 1000

# ==============================================================================
# --- DYNAMIC FILE NAMING ---
# File paths are now dynamically generated based on NUM_SIMULATIONS_TO_USE
# This prevents overwriting files for different simulation counts.
# ==============================================================================
MODEL_NAME = "MLP"
DATA_FILENAME = f"three_body_data.pt"
CHECKPOINT_PATH = f"{MODEL_NAME.lower()}_3body_checkpoint_{NUM_SIMULATIONS_TO_USE}_sims.pth"
FINAL_MODEL_PATH = f"{MODEL_NAME.lower()}_3body_final_{NUM_SIMULATIONS_TO_USE}_sims.pth"
LOSS_PLOT_PATH = f"{MODEL_NAME.lower()}_training_loss_{NUM_SIMULATIONS_TO_USE}_sims.png"
TRAJECTORY_PLOT_PATH = f"{MODEL_NAME.lower()}_trajectory_comparison_{NUM_SIMULATIONS_TO_USE}_sims.png"
COM_PLOT_PATH = f"{MODEL_NAME.lower()}_com_comparison_{NUM_SIMULATIONS_TO_USE}_sims.png"


# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training {MODEL_NAME} with {NUM_SIMULATIONS_TO_USE} simulations on device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("CUDA not available, using CPU")

# --- Load data ---
try:
    dataset = ThreeBodyDataset(filename=DATA_FILENAME, history_frames=HISTORY_FRAMES, num_sims_to_use=NUM_SIMULATIONS_TO_USE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
except ValueError as e:
    print(e)
    sys.exit(1)

# --- Model Initialization ---
INPUT_FEATURES = HISTORY_FRAMES * 12
OUTPUT_FEATURES = 12

model = MLP(input_features=INPUT_FEATURES, output_features=OUTPUT_FEATURES, hidden_size=256).to(device)

print(f"Model {MODEL_NAME} initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

start_epoch = 0
loss_history = []
best_loss = float('inf')

# --- Resume from Checkpoint ---
if os.path.exists(CHECKPOINT_PATH):
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            loss_history = checkpoint.get("loss_history", [])
            best_loss = checkpoint.get("best_loss", float('inf'))
            print(f"Resumed {MODEL_NAME} from checkpoint at epoch {start_epoch}, Best Loss: {best_loss:.6f}")
        else:
            print("Checkpoint format is incompatible. Starting fresh training.")
            os.remove(CHECKPOINT_PATH)
    except Exception as e:
        print(f"Error loading checkpoint for {MODEL_NAME}: {e}")
        print("Starting fresh training.")
        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)


# --- Training Loop ---
model.train()
print(f"\nStarting {MODEL_NAME} training...")
for epoch in range(start_epoch, NUM_EPOCHS):
    total_loss = 0
    num_batches = 0
    
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        
        pred = model(x)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    loss_history.append(avg_loss)
    
    if avg_loss < best_loss:
        best_loss = avg_loss
    
    if epoch % 50 == 0 or epoch == NUM_EPOCHS - 1:
        print(f"Epoch {epoch} | Loss: {avg_loss:.6f} | Best Loss: {best_loss:.6f}")
        
        if torch.cuda.is_available():
            print(f"GPU Memory Used: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss_history": loss_history,
            "best_loss": best_loss
        }, CHECKPOINT_PATH)
        print(f"Checkpoint saved to {CHECKPOINT_PATH}")

print(f"\n{MODEL_NAME} training completed. Final loss: {loss_history[-1]:.6f}, Best loss: {best_loss:.6f}")

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": NUM_EPOCHS,
    "loss_history": loss_history,
    "best_loss": best_loss
}, FINAL_MODEL_PATH)
print(f"Final {MODEL_NAME} model saved to {FINAL_MODEL_PATH}")

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(loss_history, 'b-', linewidth=2)
plt.title(f"{MODEL_NAME} Training Loss Over Epochs for {NUM_SIMULATIONS_TO_USE} Sims")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig(LOSS_PLOT_PATH, dpi=300, bbox_inches='tight')
plt.show()
print(f"Training loss plot saved to {LOSS_PLOT_PATH}")


# --- Evaluation and Plotting ---
print("\n" + "="*50)
print("Starting evaluation and plotting tasks...")

# Load the trained model and set to evaluation mode
model_state = torch.load(FINAL_MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(model_state["model_state_dict"])
model.eval()

# Get a single initial sample from the data (e.g., the 100th data point)
start_index = 100
if start_index + NUM_PREDICTION_STEPS + HISTORY_FRAMES >= len(dataset):
    start_index = len(dataset) - NUM_PREDICTION_STEPS - HISTORY_FRAMES - 1

start_x_norm, _, start_masses = dataset[start_index]

# Create the ground truth trajectory by unnormalizing the data
ground_truth_full_traj = []
for i in range(HISTORY_FRAMES + NUM_PREDICTION_STEPS):
    input_seq_norm, target_norm, _ = dataset[start_index + i]
    # The first HISTORY_FRAMES of the full trajectory are from the input sequence
    # The rest are the single-step target predictions
    if i < HISTORY_FRAMES:
        # Use the correct single frame from the input sequence
        frame_norm = input_seq_norm.view(HISTORY_FRAMES, 12)[i]
    else:
        # Get the target for the future step
        _, target_norm, _ = dataset[start_index + i]
        frame_norm = target_norm
    
    ground_truth_full_traj.append(dataset.unnormalize(frame_norm.to(device)))
ground_truth_full_traj = torch.stack(ground_truth_full_traj)

print(f"Model loaded successfully. Starting {NUM_PREDICTION_STEPS}-step prediction.")

# Perform multi-step prediction
with torch.no_grad():
    current_input_norm = start_x_norm.unsqueeze(0).to(device)
    predicted_sequence_norm = []
    
    for _ in range(NUM_PREDICTION_STEPS):
        # Make a prediction for the next step (in normalized space)
        predicted_next_step_norm = model(current_input_norm)
        predicted_sequence_norm.append(predicted_next_step_norm.squeeze(0))
        
        # Prepare the input for the next step by "rolling" the history
        new_input_history_norm = torch.cat([
            current_input_norm.squeeze(0)[12:].unsqueeze(0), 
            predicted_next_step_norm
        ], dim=1)
        
        current_input_norm = new_input_history_norm
        
predicted_sequence_norm = torch.stack(predicted_sequence_norm)

# Combine the unnormalized history with the predicted sequence to form the full trajectory
start_history_states_unnorm = dataset.unnormalize(start_x_norm).view(HISTORY_FRAMES, 12).to(device)
predicted_sequence_unnorm = dataset.unnormalize(predicted_sequence_norm)
predicted_full_traj = torch.cat([start_history_states_unnorm, predicted_sequence_unnorm], dim=0)

print("Multi-step prediction complete.")

# Plot the trajectories
plot_trajectory(predicted_full_traj, ground_truth_full_traj, NUM_PREDICTION_STEPS, TRAJECTORY_PLOT_PATH, NUM_SIMULATIONS_TO_USE)

# Calculate and plot the Center of Mass (COM)
masses = start_masses.to(device)

predicted_com = calculate_center_of_mass(predicted_full_traj, masses)
ground_truth_com = calculate_center_of_mass(ground_truth_full_traj, masses)

plot_com_trajectory(predicted_com, ground_truth_com, COM_PLOT_PATH, NUM_SIMULATIONS_TO_USE)

print("Evaluation script finished.")
print("="*50)
