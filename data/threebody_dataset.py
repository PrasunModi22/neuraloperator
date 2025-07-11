import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def simulate_3body(num_steps=100, dt=0.01, seed=42):
    np.random.seed(seed)
    N = 3     
    dim = 2    
    G = 1.0
    masses = np.ones(N)

    pos = np.random.uniform(-1, 1, size=(N, dim))
    vel = np.random.uniform(-0.1, 0.1, size=(N, dim))

    trajectory = np.zeros((num_steps, N * dim * 2), dtype=np.float32)

    for t in range(num_steps):
        acc = np.zeros((N, dim), dtype=np.float32)
        for a in range(N):
            for b in range(N):
                if a != b:
                    r_vec = pos[b] - pos[a]
                    r = np.linalg.norm(r_vec)
                    if r > 1e-6:
                        acc[a] += G * masses[b] * r_vec / (r**3)

        vel += acc * dt
        pos += vel * dt
        trajectory[t] = np.hstack((pos.flatten(), vel.flatten()))
    
    return trajectory[np.newaxis, :, :] 

    np.random.seed(seed)
    N = 3       
    dim = 2     
    G = 1.0
    masses = np.ones(N)
    pos = np.random.uniform(-1, 1, size=(N, dim))
    vel = np.random.uniform(-0.1, 0.1, size=(N, dim))
    trajectory = np.zeros((num_steps, N * dim * 2), dtype=np.float32)
    for t in range(num_steps):
        acc = np.zeros((N, dim), dtype=np.float32)
        for a in range(N):
            for b in range(N):
                if a != b:
                    r_vec = pos[b] - pos[a]
                    r = np.linalg.norm(r_vec)
                    if r > 1e-6:
                        acc[a] += G * masses[b] * r_vec / (r**3)
        vel += acc * dt
        pos += vel * dt
        trajectory[t] = np.hstack((pos.flatten(), vel.flatten()))
    return trajectory[np.newaxis, :, :]

data = simulate_3body()
np.savez("threebody_single.npz", data=data)


class ThreeBodyNormalizedDataset(Dataset):
    def __init__(self, filepath, rollout=1):
        raw = np.load(filepath)["data"]  
        self.data = torch.from_numpy(raw).float()
        self.data_min = self.data.min(dim=1, keepdim=True)[0]
        self.data_max = self.data.max(dim=1, keepdim=True)[0]
        self.normalized_data = (self.data - self.data_min) / (self.data_max - self.data_min + 1e-8)
        self.rollout = rollout
        self.samples = []
        sim, T, _ = self.normalized_data.shape
        for t in range(T - rollout):
            x = self.normalized_data[0, t]
            y = self.normalized_data[0, t + rollout]
            self.samples.append((x, y))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]


dataset = ThreeBodyNormalizedDataset("threebody_single.npz", rollout=1)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

for x, y in loader:
    print("Input:", x)
    print("Target:", y)
    break