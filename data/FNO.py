import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import sys
from data import ThreeBodyDataset

# FNO Model
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        modes_to_keep = min(self.modes, x.shape[2] // 2 + 1)

        fourier_x = torch.fft.rfft(x, dim=2, norm="forward")

        out_fft = torch.zeros(x.shape[0], self.out_channels, x.shape[2]//2 + 1, device=x.device, dtype=torch.cfloat)
        
        out_fft[:, :, :modes_to_keep] = self.compl_mul1d(fourier_x[:, :, :modes_to_keep], self.weights1)
        
        x_out = torch.fft.irfft(out_fft, n=x.shape[2], dim=2, norm="forward")
        
        return x_out


class FNO(nn.Module):
    def __init__(self, input_features, output_features, width, modes):
        super(FNO, self).__init__()
        
        self.input_features = input_features
        self.output_features = output_features
        self.width = width
        self.modes = modes
        
        self.lifting_conv = nn.Conv1d(1, width, 1)

        self.spectral_conv1 = SpectralConv1d(width, width, modes)
        self.skip_conv1 = nn.Conv1d(width, width, 1)
        
        self.spectral_conv2 = SpectralConv1d(width, width, modes)
        self.skip_conv2 = nn.Conv1d(width, width, 1)
        
        self.projection_conv = nn.Conv1d(width, 1, 1)
        
        self.fc1 = nn.Linear(input_features, 128)
        self.fc2 = nn.Linear(128, output_features)
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        original_shape = x.shape
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = x.unsqueeze(1) 
        
        x = F.gelu(self.lifting_conv(x))
        
        res = x 
        x = self.spectral_conv1(x)
        x = F.gelu(x + self.skip_conv1(res)) 
        x = self.dropout(x)
        
        res = x
        x = self.spectral_conv2(x)
        x = F.gelu(x + self.skip_conv2(res))
        x = self.dropout(x)
        
        x = self.projection_conv(x)
        x = x.squeeze(1) 
        
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) 
        
        if len(original_shape) == 1:
            x = x.squeeze(0)
            
        return x

# Training

HISTORY_FRAMES = 2
BATCH_SIZE = 64  
NUM_EPOCHS = 500  
LEARNING_RATE = 1e-3

MODEL_NAME = "FNO"
CHECKPOINT_PATH = f"{MODEL_NAME.lower()}_3body_checkpoint.pth"
FINAL_MODEL_PATH = f"{MODEL_NAME.lower()}_3body_final.pth"
LOSS_PLOT_PATH = f"{MODEL_NAME.lower()}_training_loss.png"
EVAL_PLOT_PATH = f"{MODEL_NAME.lower()}_evaluation_plot.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training {MODEL_NAME} on device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("CUDA not available, using CPU")

dataset = ThreeBodyDataset(filename="three_body_data.pt", history_frames=HISTORY_FRAMES)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

INPUT_FEATURES = HISTORY_FRAMES * 12
OUTPUT_FEATURES = 12
model = FNO(input_features=INPUT_FEATURES, output_features=OUTPUT_FEATURES, width=64, modes=4).to(device)

print(f"Model {MODEL_NAME} initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

start_epoch = 0
loss_history = []
best_loss = float('inf')

if os.path.exists(CHECKPOINT_PATH):
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        
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

model.train()
print(f"\nStarting {MODEL_NAME} training...")
for epoch in range(start_epoch, NUM_EPOCHS):
    total_loss = 0
    num_batches = 0

    for x, y in loader:
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

