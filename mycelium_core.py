"""
Mycelium AI: Official Implementation
Author: Cássio Rodrigues Alves (2026)

This script implements the Mycelium Diffusion V6 architecture with:
1. Stone-Sand Duality
2. Meta-Nerve Mechanism (with Homeostatic Calibration)
3. Sinusoidal Time Embeddings & Graph Topology
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. CORE ARCHITECTURE COMPONENTS
# ==========================================

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class MycelialBlock(nn.Module):
    """
    The fundamental unit of Mycelium AI.
    Implements the Stone (Fixed) and Sand (Plastic) interaction.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_stone = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim) 
        
        # Plasticity Hyperparameters (from Paper Table 1)
        self.gamma = 0.90 
        self.eta = 0.15   

    def forward(self, x, adj_norm):
        # 1. Biological Message Passing (Stone Path)
        m = self.norm(torch.einsum('ij, bjc -> bic', adj_norm, x))
        m = m + x  
        m_processed = F.gelu(self.W_stone(m))
        
        # 2. Meta-Nerve Mechanism (Uncertainty Detection)
        # Calculates geometric divergence
        local_var = torch.abs(m_processed - x).mean(dim=-1, keepdim=True)
        
        # Homeostatic Calibration (Temperature Scaling) - The "Cure"
        # T=2.0, Threshold=1.0 to prevent anxiety/saturation
        g_meta = torch.sigmoid((local_var - 1.0) / 2.0)
        
        # 3. Sand Memory Update (Hebbian Rule)
        # Note: In this implementation, Sand is transient for the forward pass context
        w_sand = torch.zeros_like(x) # Initialized at 0 for the step
        correlation = x * m_processed
        
        # Update rule: W_new = gamma * W_old + eta * gate * hebbian
        w_sand = torch.clamp(self.gamma * w_sand + self.eta * g_meta * correlation, -1.5, 1.5)
        
        # 4. Final Hyphal Update
        out = x + (w_sand * m_processed)
        return out, g_meta

class MyceliumDiffusion(nn.Module):
    def __init__(self, num_nodes, num_classes, hidden_dim=128): 
        super().__init__()
        self.num_nodes = num_nodes
        
        # Node Consciousness (Inputs)
        self.proj_in = nn.Linear(1, hidden_dim)
        self.node_embed = nn.Embedding(num_nodes, hidden_dim)      
        self.class_embed = nn.Embedding(num_classes, hidden_dim)
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU()
        )
        
        # Deep Graph Stack
        self.layers = nn.ModuleList([MycelialBlock(hidden_dim) for _ in range(4)])
        self.proj_out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, x, t, class_labels, adjacency):
        # Initialize State
        h = self.proj_in(x.unsqueeze(-1)) 
        h = h + self.time_mlp(t).unsqueeze(1) 
        h = h + self.node_embed(torch.arange(self.num_nodes, device=x.device)).unsqueeze(0)
        h = h + self.class_embed(class_labels).unsqueeze(1)
        
        meta_nerve_activity = []
        
        for layer in self.layers:
            h, g = layer(h, adjacency)
            meta_nerve_activity.append(g)
            
        return self.proj_out(h).squeeze(-1), meta_nerve_activity

# ==========================================
# 2. UTILS & GRAPH TOPOLOGY
# ==========================================

def create_small_world_graph(grid_size=16):
    """Creates a normalized adjacency matrix with small-world properties."""
    N = grid_size * grid_size
    A = torch.zeros((N, N))
    for i in range(N):
        x, y = i % grid_size, i // grid_size
        if x > 0: A[i, i - 1] = 1
        if x < grid_size - 1: A[i, i + 1] = 1
        if y > 0: A[i, i - grid_size] = 1
        if y < grid_size - 1: A[i, i + grid_size] = 1
        
    # Add random long-range connections (Hyphal shortcuts)
    A = torch.clamp(A + (torch.rand(N, N) < 0.05).float(), 0, 1) 
    A = A + torch.eye(N) * 2.0 # Strong self-connection
    
    # Laplacian Normalization
    degree = A.sum(dim=-1, keepdim=True)
    return A / (degree + 1e-8)

def get_ddpm_schedule(timesteps=100):
    beta = torch.linspace(0.0001, 0.02, timesteps)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_bar

# ==========================================
# 3. DEMO EXECUTION (If run directly)
# ==========================================
if __name__ == "__main__":
    print("🍄 Mycelium AI V6 Core Loaded.")
    print("To train and generate, please check the Notebooks in the repo.")
    
    # Simple sanity check
    model = MyceliumDiffusion(num_nodes=256, num_classes=2)
    adj = create_small_world_graph(16)
    x = torch.randn(2, 256)
    t = torch.tensor([10, 10])
    c = torch.tensor([0, 1])
    
    out, activity = model(x, t, c, adj)
    print(f"Forward Pass Successful. Output Shape: {out.shape}")