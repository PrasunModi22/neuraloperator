import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random

def simulate_3body(num_steps=500, dt=0.01, min_distance=0.1):
    def grav_force(pos_i, pos_j, mass_j):
        r = pos_j - pos_i
        dist = torch.norm(r) + 1e-5
        return r * mass_j / (dist**3)

    positions = torch.rand(3, 2) * 2 - 1 
    velocities = torch.randn(3, 2) * 0.1
    masses = torch.tensor([1.0, 1.0, 1.0])
    trajectory = []

    for _ in range(num_steps):
        forces = torch.zeros(3, 2)
        for i in range(3):
            for j in range(3):
                if i != j:
                    dist = torch.norm(positions[i] - positions[j])
                    if dist < min_distance:
                        
                        direction = (positions[i] - positions[j])
                        forces[i] += direction / (dist**2 + 1e-5)
                    else:
                        forces[i] += grav_force(positions[i], positions[j], masses[j])
        accelerations = forces / masses.unsqueeze(1)
        velocities += accelerations * dt
        positions += velocities * dt
        state = torch.cat([positions.flatten(), velocities.flatten()])
        trajectory.append(state.unsqueeze(0))

    trajectory = torch.cat(trajectory, dim=0).unsqueeze(0) 
    return trajectory.numpy()

# Save to disk
data = simulate_3body()
np.savez("threebody_single.npz", data=data)

class NBodyTrajectoryDataset(Dataset):
    def __init__(self, filepath: str, rollout: int = 1, normalization_strategy: str = 'per_simulation'):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found at {filepath}")
        if rollout < 1:
            raise ValueError("Rollout must be at least 1.")

        with np.load(filepath) as f:
            self.raw_data = f["data"].astype(np.float32)

        self.rollout = rollout
        self.num_simulations, self.timesteps, self.num_features = self.raw_data.shape

        self.normalization_strategy = normalization_strategy
        self._setup_normalization()

        self.num_samples_per_sim = self.timesteps - self.rollout
        self.total_samples = self.num_simulations * self.num_samples_per_sim

    def _setup_normalization(self):
        if self.normalization_strategy == 'global':
            self.data_min = np.min(self.raw_data, axis=(0, 1), keepdims=True)
            self.data_max = np.max(self.raw_data, axis=(0, 1), keepdims=True)
        elif self.normalization_strategy == 'per_simulation':
            self.data_min = np.min(self.raw_data, axis=1, keepdims=True)
            self.data_max = np.max(self.raw_data, axis=1, keepdims=True)
        else:
            self.data_min = self.data_max = None

    def normalize(self, data: np.ndarray, sim_idx: int) -> np.ndarray:
        if self.data_min is None:
            return data
        min_vals = self.data_min if self.normalization_strategy == 'global' else self.data_min[sim_idx]
        max_vals = self.data_max if self.normalization_strategy == 'global' else self.data_max[sim_idx]
        return (data - min_vals) / (max_vals - min_vals + 1e-8)

    def unnormalize(self, normalized_data: torch.Tensor, sim_idx: int) -> torch.Tensor:
        if self.data_min is None:
            return normalized_data
        min_vals = torch.tensor(self.data_min[sim_idx], device=normalized_data.device)
        max_vals = torch.tensor(self.data_max[sim_idx], device=normalized_data.device)
        return normalized_data * (max_vals - min_vals + 1e-8) + min_vals

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int):
        sim_idx = idx // self.num_samples_per_sim
        time_idx = idx % self.num_samples_per_sim
        x_raw = self.raw_data[sim_idx, time_idx]
        y_raw = self.raw_data[sim_idx, time_idx + self.rollout]
        x = torch.from_numpy(self.normalize(x_raw, sim_idx))
        y = torch.from_numpy(self.normalize(y_raw, sim_idx))
        return x, y, sim_idx

dataset = NBodyTrajectoryDataset("threebody_single.npz", rollout=1)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

class SimpleMLP(nn.Module):
    def __init__(self, width, in_features):
        super().__init__()
        self.width = width
        self.fc0 = nn.Linear(in_features, width)
        
        self.mlp1 = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width)
        )
        
        self.mlp2 = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width)
        )
        
        self.w1 = nn.Linear(width, width)
        self.w2 = nn.Linear(width, width)
        
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, in_features)
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        original_shape = x.shape
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = self.fc0(x)
    
        x1 = self.mlp1(x)
        x2 = self.w1(x)
        x = F.relu(x1 + x2)
        x = self.dropout(x)
        
        x1 = self.mlp2(x)
        x2 = self.w2(x)
        x = F.relu(x1 + x2)
        x = self.dropout(x)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        if len(original_shape) == 1:
            x = x.squeeze(0)
        
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Initial GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
else:
    print("CUDA not available, using CPU")

use_amp = torch.cuda.is_available() and hasattr(torch.cuda, 'amp')
if use_amp:
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    print("Using mixed precision training (AMP)")

model = SimpleMLP(width=64, in_features=12).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
checkpoint_path = "mlp_3body.pth"
start_epoch = 0
loss_history = []
best_loss = float('inf')

if os.path.exists(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            loss_history = checkpoint.get("loss_history", [])
            best_loss = checkpoint.get("best_loss", float('inf'))
            print(f"Resumed from checkpoint at epoch {start_epoch}")
        else:
            print("Checkpoint format is incompatible. Starting fresh training.")
            os.remove(checkpoint_path)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Starting fresh training.")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

model.train()
for epoch in range(start_epoch, 500):
    total_loss = 0
    num_batches = 0
    
    for x, y, sim_idx in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        if use_amp:
            with autocast():
                pred = model(x)
                loss = loss_fn(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    loss_history.append(avg_loss)
    
    # Track best loss
    if avg_loss < best_loss:
        best_loss = avg_loss
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Loss: {avg_loss:.6f} | Best: {best_loss:.6f}")
        
        if torch.cuda.is_available():
            print(f"GPU Memory Used: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss_history": loss_history,
            "best_loss": best_loss
        }, checkpoint_path)

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": 500,
    "loss_history": loss_history,
    "best_loss": best_loss
}, "mlp_3body_final.pth")

print(f"Training completed. Final loss: {loss_history[-1]:.6f}, Best loss: {best_loss:.6f}")

plt.figure(figsize=(10, 6))
plt.plot(loss_history, 'b-', linewidth=2)
plt.title("MLP Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.yscale('log') 
plt.tight_layout()
plt.savefig('mlp_training_loss.png', dpi=300, bbox_inches='tight')
plt.show()

model.eval()
x, y, sim_idx = dataset[0]
x = x.to(device)
with torch.no_grad():
    pred = model(x)

gt = dataset.unnormalize(y, sim_idx)
out = dataset.unnormalize(pred.cpu(), sim_idx)

fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
axs[0].plot(gt.numpy(), label="Ground Truth")
axs[0].legend(); axs[0].set_title("Ground Truth")

axs[1].plot(out.numpy(), label="Prediction")
axs[1].legend(); axs[1].set_title("MLP Prediction")

axs[2].plot((gt - out).numpy(), label="Error")
axs[2].legend(); axs[2].set_title("Prediction Error")

plt.tight_layout()
plt.show()