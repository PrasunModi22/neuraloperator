import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import sys
from matplotlib.lines import Line2D

# Import FNO2d from the neuralop library
try:
    from neuralop.models import FNO2d
except ImportError:
    print("Error: The 'neuralop' library is not installed.")
    print("Please install it using: pip install neuralop")
    sys.exit(1)

# Assuming the data.py file is in the same directory and contains the ThreeBodyDataset
# class.
try:
    from data import ThreeBodyDataset
except ImportError:
    print("Error: Could not import ThreeBodyDataset from data.py.")
    print("Please ensure data.py is in the same directory.")
    sys.exit(1)


# ==============================================================================
# --- FNO Model Wrapper ---
# ==============================================================================
class FNO2d_Wrapper(nn.Module):
    """
    A wrapper for the FNO2d model that handles reshaping the 1D input vector
    into a 2D grid suitable for the FNO, and then flattens the output.
    """
    def __init__(self, in_features, out_features, hidden_channels, modes_height, modes_width):
        super().__init__()
        
        # We'll reshape the input of size `in_features` into a 2D grid
        # For a history of 2 frames (2 * 12 = 24), we can reshape to (2, 12).
        self.height = 2
        self.width = 12
        assert in_features == self.height * self.width, "Input features must match the reshape dimensions."

        # Initialize the FNO2d model from neuralop
        # The input to FNO is [batch, channels, height, width]
        self.fno_model = FNO2d(
            n_modes_height=modes_height,
            n_modes_width=modes_width,
            in_channels=1,  # We treat the input as a single channel grid
            out_channels=1,
            hidden_channels=hidden_channels
        )
        
        # Add a final linear layer to project the FNO output to the desired output features
        fno_output_size = self.height * self.width
        self.fc = nn.Linear(fno_output_size, out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
        """
        # Reshape the input tensor to a 2D grid with a channel dimension
        # Input shape: [batch_size, in_features]
        # Reshaped shape: [batch_size, 1, height, width]
        x = x.view(x.size(0), 1, self.height, self.width)
        
        # Pass through the FNO2d model
        x = self.fno_model(x)
        
        # Flatten the FNO output and pass through the linear layer
        # Output from FNO is [batch, 1, height, width]. Flatten from dimension 1.
        x = x.flatten(start_dim=1)
        
        # Final output shape: [batch_size, out_features]
        x = self.fc(x)
        return x


# ==============================================================================
# --- Helper Functions for Evaluation and Plotting ---
# ==============================================================================

def calculate_center_of_mass(states, masses):
    """
    Calculates the center of mass for a set of three-body states.
    states: torch.Tensor of shape [num_steps, 12]
    masses: torch.Tensor of shape [3]
    Returns: torch.Tensor of shape [num_steps, 2]
    """
    positions = states.view(-1, 3, 4)[..., :2] # Corrected to get x,y positions only
    total_mass = masses.sum()
    weighted_positions = positions * masses.view(1, 3, 1)
    com_positions = weighted_positions.sum(dim=1) / total_mass
    return com_positions


def plot_trajectory(predicted_traj, ground_truth_traj, plot_path, num_steps):
    """
    Plots the predicted and ground truth 2D trajectories of the three bodies.
    """
    plt.figure(figsize=(10, 8))
    
    pred_pos = predicted_traj.view(-1, 3, 4)
    gt_pos = ground_truth_traj.view(-1, 3, 4)
    
    colors = ['r', 'g', 'b']
    labels = ['Body 1', 'Body 2', 'Body 3']
    
    # Plot predicted trajectories
    for i in range(3):
        plt.plot(pred_pos[:, i, 0], pred_pos[:, i, 1], '--', color=colors[i], label=f'Predicted {labels[i]}')
        
    # Plot ground truth trajectories
    for i in range(3):
        plt.plot(gt_pos[:, i, 0], gt_pos[:, i, 1], '-', color=colors[i], label=f'Ground Truth {labels[i]}')
        
    plt.title(f"Predicted vs. Ground Truth Trajectory (Steps 1 to {num_steps})")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Trajectory plot saved to {plot_path}")


def plot_com_trajectory(predicted_com, ground_truth_com, plot_path):
    """
    Plots the center of mass trajectory for predictions and ground truth.
    """
    plt.figure(figsize=(10, 8))
    
    plt.plot(predicted_com[:, 0], predicted_com[:, 1], 'ro-', label='Predicted COM')
    plt.plot(ground_truth_com[:, 0], ground_truth_com[:, 1], 'bo-', label='Ground Truth COM')
    
    plt.title("Center of Mass Trajectory")
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
NUM_SIMULATIONS_TO_USE = 10

# File paths
MODEL_NAME = "neuralop_FNO2d"
CHECKPOINT_PATH = f"{MODEL_NAME.lower()}_3body_checkpoint.pth"
FINAL_MODEL_PATH = f"{MODEL_NAME.lower()}_3body_final.pth"
LOSS_PLOT_PATH = f"{MODEL_NAME.lower()}_training_loss.png"
TRAJECTORY_PLOT_PATH = "fno_trajectory_comparison.png"
COM_PLOT_PATH = "fno_com_comparison.png"

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training {MODEL_NAME} on device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("CUDA not available, using CPU")


# --- Data Loading ---
try:
    # This is the key change: use the updated dataset directly
    dataset = ThreeBodyDataset(filename="three_body_data.pt", history_frames=HISTORY_FRAMES, num_sims_to_use=NUM_SIMULATIONS_TO_USE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
except ValueError as e:
    print(e)
    sys.exit(1)


# --- Model Initialization ---
INPUT_FEATURES = HISTORY_FRAMES * 12
OUTPUT_FEATURES = 12

model = FNO2d_Wrapper(
    in_features=INPUT_FEATURES,
    out_features=OUTPUT_FEATURES,
    hidden_channels=64,
    modes_height=2,
    modes_width=6
).to(device)

print(f"Model {MODEL_NAME} initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

start_epoch = 0
loss_history = []
best_loss = float('inf')

# --- Resume from Checkpoint ---
if os.path.exists(CHECKPOINT_PATH):
    try:
        # NOTE: Added weights_only=False to support loading older models on newer PyTorch versions
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

    for x, y, _ in loader:  # Note: `_` is used to ignore the masses
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
plt.title(f"{MODEL_NAME} Training Loss Over Epochs")
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
# NOTE: Added weights_only=False to support loading older models on newer PyTorch versions
model_state = torch.load(FINAL_MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(model_state["model_state_dict"])
model.eval()

# Get a single initial sample from the data (e.g., the 100th data point)
start_index = 100
if start_index + NUM_PREDICTION_STEPS + HISTORY_FRAMES >= len(dataset):
    start_index = len(dataset) - NUM_PREDICTION_STEPS - HISTORY_FRAMES - 1

start_x, _, start_masses = dataset[start_index]

# Create the ground truth sequence for comparison
# The dataset returns normalized data, so we must unnormalize for plotting
ground_truth_sequence = []
for i in range(NUM_PREDICTION_STEPS):
    _, ground_truth_y_norm, _ = dataset[start_index + HISTORY_FRAMES + i]
    ground_truth_sequence.append(dataset.unnormalize(ground_truth_y_norm.to(device)))

ground_truth_sequence = torch.stack(ground_truth_sequence)

# Get the initial history frames for the ground truth trajectory
gt_history_frames = []
for i in range(HISTORY_FRAMES):
    input_frame, _, _ = dataset[start_index + i]
    gt_history_frames.append(dataset.unnormalize(input_frame[i*12:(i+1)*12]))
gt_history_frames = torch.stack(gt_history_frames).to(device)

ground_truth_full_traj = torch.cat([gt_history_frames, ground_truth_sequence], dim=0)

print(f"Model loaded successfully. Starting {NUM_PREDICTION_STEPS}-step prediction.")

# Perform multi-step prediction
with torch.no_grad():
    current_input_norm = start_x.unsqueeze(0).to(device)  # Add batch dimension
    predicted_sequence_norm = []
    
    for i in range(NUM_PREDICTION_STEPS):
        # Make a prediction for the next step
        predicted_next_step_norm = model(current_input_norm)
        predicted_sequence_norm.append(predicted_next_step_norm.squeeze(0))
        
        # Prepare the input for the next step by "rolling" the history
        new_input_history_norm = torch.cat([
            current_input_norm.squeeze(0)[12:].unsqueeze(0), 
            predicted_next_step_norm
        ], dim=1)
        
        current_input_norm = new_input_history_norm
        
predicted_sequence_norm = torch.stack(predicted_sequence_norm)
predicted_sequence_unnorm = dataset.unnormalize(predicted_sequence_norm)

# Combine the history with the predicted sequence to form the full trajectory
start_history_states_unnorm = dataset.unnormalize(start_x.view(HISTORY_FRAMES, 12).to(device))
predicted_full_traj = torch.cat([start_history_states_unnorm, predicted_sequence_unnorm], dim=0)

print("Multi-step prediction complete.")

# Plot the trajectories
plot_trajectory(predicted_full_traj, ground_truth_full_traj, TRAJECTORY_PLOT_PATH, NUM_PREDICTION_STEPS)

# Calculate and plot the Center of Mass (COM)
masses = start_masses.to(device)

predicted_com = calculate_center_of_mass(predicted_full_traj, masses)
ground_truth_com = calculate_center_of_mass(ground_truth_full_traj, masses)

plot_com_trajectory(predicted_com.cpu(), ground_truth_com.cpu(), COM_PLOT_PATH)

print("Evaluation script finished.")
print("="*50)
