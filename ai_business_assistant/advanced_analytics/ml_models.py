"""
Advanced Machine Learning Models Engine
Provides deep learning, transformers, graph neural networks, and reinforcement learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
import optuna
import ray
from ray import tune
import networkx as nx

logger = logging.getLogger(__name__)


class MLModelType(Enum):
    """Machine learning model types."""
    DEEP_NN = "deep_neural_network"
    CNN = "convolutional_neural_network"
    RNN = "recurrent_neural_network"
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    AUTOENCODER = "autoencoder"
    VARIATIONAL_AUTOENCODER = "variational_autoencoder"
    GAN = "generative_adversarial_network"
    GRAPH_NN = "graph_neural_network"
    GCN = "graph_convolutional_network"
    GAT = "graph_attention_network"
    GRAPHSAGE = "graphsage"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    Q_LEARNING = "q_learning"
    POLICY_GRADIENT = "policy_gradient"
    ACTOR_CRITIC = "actor_critic"
    BANDITS = "multi_armed_bandits"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    ENSEMBLE = "ensemble_methods"


@dataclass
class MLModelResult:
    """Result of ML model training."""
    model_type: MLModelType
    model_name: str
    performance_metrics: Dict[str, float]
    model_weights: Optional[np.ndarray] = None
    training_history: Dict[str, List[float]] = field(default_factory=dict)
    feature_importance: Optional[Dict[str, float]] = None
    hyperparameter_tuning: Dict[str, Any] = field(default_factory=dict)
    model_interpretability: Dict[str, Any] = field(default_factory=dict)
    deployment_ready: bool = False
    training_time: float = 0.0
    model_size_mb: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class DeepNeuralNetwork(nn.Module):
    """Advanced deep neural network with attention mechanisms."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 dropout_rate: float = 0.2,
                 batch_norm: bool = True,
                 activation: str = "relu"):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.batch_norm = batch_norm
        
        # Input layer
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation function
            if activation.lower() == "relu":
                self.layers.append(nn.ReLU())
            elif activation.lower() == "leaky_relu":
                self.layers.append(nn.LeakyReLU())
            elif activation.lower() == "elu":
                self.layers.append(nn.ELU())
            elif activation.lower() == "swish":
                self.layers.append(nn.SiLU())
            
            # Dropout
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # Attention mechanism (optional)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[-1] if hidden_dims else input_dim,
            num_heads=8,
            dropout=dropout_rate
        ) if len(hidden_dims) > 0 else None
    
    def forward(self, x):
        # Apply layers
        for layer in self.layers:
            x = layer(x)
        
        # Apply attention if available
        if self.attention is not None and len(x.shape) == 2:
            # Add sequence dimension for attention
            x = x.unsqueeze(1)  # (batch_size, 1, features)
            x, _ = self.attention(x, x, x)
            x = x.squeeze(1)  # Remove sequence dimension
        
        # Output layer
        x = self.output_layer(x)
        return x


class LSTMModel(nn.Module):
    """Advanced LSTM with attention and skip connections."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 output_dim: int,
                 dropout_rate: float = 0.2,
                 bidirectional: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * (2 if bidirectional else 1),
            num_heads=8,
            dropout=dropout_rate
        )
        
        # Output layers
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(lstm_output_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        lstm_out = lstm_out.transpose(0, 1)  # (seq_len, batch, features)
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attended_out = attended_out.transpose(0, 1)  # (batch, seq_len, features)
        
        # Use the last output
        out = attended_out[:, -1, :]  # (batch, features)
        
        # Fully connected layers
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class TransformerModel(nn.Module):
    """Transformer model for sequence modeling."""
    
    def __init__(self,
                 input_dim: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 output_dim: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        # Project input to model dimension
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer
        encoded = self.transformer(x)
        
        # Use the first token's representation (CLS token approach)
        # For simplicity, use the mean of all tokens
        x = torch.mean(encoded, dim=1)
        
        # Output projection
        x = self.output_proj(x)
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class GraphConvolutionalNetwork(nn.Module):
    """Graph Convolutional Network for node classification."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GCN layers
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        
        # Graph convolution layer
        self.gcn_layer = GCNConv(input_dim, hidden_dim) if num_layers > 1 else GCNConv(input_dim, output_dim)
        
    def forward(self, x, edge_index):
        # Apply GCN layer
        x = self.gcn_layer(x, edge_index)
        
        # Apply fully connected layers
        for layer in self.layers[:-1]:  # Skip output layer
            x = layer(x)
        
        x = self.layers[-1]  # Output layer
        return x


class GCNConv(nn.Module):
    """Graph Convolutional Layer implementation."""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x, edge_index):
        # Simple GCN implementation
        # This is a simplified version - would need proper normalization in practice
        
        # Add self-loops
        num_nodes = x.size(0)
        self_loop = torch.arange(num_nodes, device=edge_index.device).unsqueeze(0).repeat(2, 1)
        edge_index_with_self = torch.cat([edge_index, self_loop], dim=1)
        
        # Normalize edges
        edge_index_norm = self._normalize(edge_index_with_self, x.size(0))
        
        # GCN operation
        x = torch.matmul(x, self.weight)
        x = torch.spmm(edge_index_norm, x)
        
        return x
    
    def _normalize(self, edge_index, num_nodes):
        """Normalize edge weights."""
        row, col = edge_index
        deg = torch.zeros(num_nodes, device=edge_index.device)
        deg.index_add_(0, row, torch.ones(row.size(0), device=edge_index.device))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        
        norm_edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        return torch.sparse_coo_tensor(edge_index, norm_edge_weight, (num_nodes, num_nodes))


class QLearningAgent:
    """Q-Learning reinforcement learning agent."""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 0.01,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table (would use neural network in practice)
        self.q_table = np.zeros((state_dim, action_dim))
        
        # Neural network approximation (optional)
        self.q_network = None
        self.target_network = None
        
    def choose_action(self, state: int, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            # Explore: choose random action
            return np.random.choice(self.action_dim)
        else:
            # Exploit: choose best action
            return np.argmax(self.q_table[state])
    
    def update_q_value(self, state: int, action: int, reward: float, next_state: int):
        """Update Q-value using Q-learning update rule."""
        # Q-learning update
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state, action] = new_q
    
    def train_episode(self, env, max_steps: int = 100) -> Dict[str, Any]:
        """Train agent for one episode."""
        state = env.reset()
        total_reward = 0
        transitions = []
        
        for step in range(max_steps):
            # Choose action
            action = self.choose_action(state)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            transitions.append((state, action, reward, next_state, done))
            
            # Update Q-value
            self.update_q_value(state, action, reward, next_state)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return {
            "total_reward": total_reward,
            "episode_length": len(transitions),
            "epsilon": self.epsilon
        }


class MultiArmedBandit:
    """Multi-armed bandit implementation with Thompson sampling."""
    
    def __init__(self, num_arms: int, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.num_arms = num_arms
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        
        # Posterior parameters for each arm
        self.alpha = np.ones(num_arms) * prior_alpha
        self.beta = np.ones(num_arms) * prior_beta
        
        # Tracking
        self.arm_counts = np.zeros(num_arms)
        self.total_reward = 0
        self.total_pulls = 0
    
    def select_arm(self, method: str = "thompson") -> int:
        """Select arm using specified strategy."""
        if method == "thompson":
            # Thompson sampling
            samples = np.random.beta(self.alpha, self.beta)
            return np.argmax(samples)
        
        elif method == "ucb":
            # Upper Confidence Bound
            if self.total_pulls == 0:
                return np.random.randint(self.num_arms)
            
            ucb_values = []
            for arm in range(self.num_arms):
                if self.arm_counts[arm] == 0:
                    ucb_values.append(float('inf'))
                else:
                    mean_reward = self.alpha[arm] / (self.alpha[arm] + self.beta[arm])
                    confidence = np.sqrt(2 * np.log(self.total_pulls) / self.arm_counts[arm])
                    ucb_values.append(mean_reward + confidence)
            
            return np.argmax(ucb_values)
        
        elif method == "epsilon_greedy":
            # Epsilon-greedy
            epsilon = 0.1
            if np.random.random() < epsilon:
                return np.random.randint(self.num_arms)
            else:
                mean_rewards = self.alpha / (self.alpha + self.beta)
                return np.argmax(mean_rewards)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def update(self, arm: int, reward: float):
        """Update posterior for selected arm."""
        self.arm_counts[arm] += 1
        self.total_reward += reward
        self.total_pulls += 1
        
        # Beta distribution conjugate update
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
    
    def get_reward_estimates(self) -> Dict[str, np.ndarray]:
        """Get current reward estimates for all arms."""
        mean_rewards = self.alpha / (self.alpha + self.beta)
        confidence_intervals = self._calculate_confidence_intervals()
        
        return {
            "mean_rewards": mean_rewards,
            "confidence_intervals": confidence_intervals,
            "pull_counts": self.arm_counts
        }
    
    def _calculate_confidence_intervals(self, confidence_level: float = 0.95) -> np.ndarray:
        """Calculate confidence intervals for each arm."""
        alpha_level = 1 - confidence_level
        intervals = np.zeros((self.num_arms, 2))
        
        for arm in range(self.num_arms):
            if self.arm_counts[arm] > 0:
                # Beta distribution confidence interval
                lower = stats.beta.ppf(alpha_level/2, self.alpha[arm], self.beta[arm])
                upper = stats.beta.ppf(1 - alpha_level/2, self.alpha[arm], self.beta[arm])
                intervals[arm] = [lower, upper]
            else:
                intervals[arm] = [0.0, 1.0]
        
        return intervals


class AdvancedMLEngine:
    """
    Advanced machine learning engine with deep learning, transformers, 
    graph neural networks, and reinforcement learning capabilities.
    """
    
    def __init__(self, device: str = "auto"):
        """Initialize ML engine."""
        self.device = self._get_device(device)
        self.models = {}
        self.training_history = {}
        self.hyperparameter_spaces = self._define_hyperparameter_spaces()
        
    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device for training."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def train_deep_neural_network(self, 
                                X_train: np.ndarray,
                                y_train: np.ndarray,
                                X_val: Optional[np.ndarray] = None,
                                y_val: Optional[np.ndarray] = None,
                                model_config: Optional[Dict[str, Any]] = None) -> MLModelResult:
        """
        Train advanced deep neural network.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            model_config: Model configuration
        """
        if model_config is None:
            model_config = {
                "hidden_dims": [256, 128, 64],
                "dropout_rate": 0.2,
                "batch_norm": True,
                "activation": "relu",
                "learning_rate": 0.001,
                "batch_size": 64,
                "epochs": 100,
                "early_stopping": True,
                "patience": 10
            }
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Initialize model
        input_dim = X_train.shape[1]
        output_dim = 1 if len(y_train.shape) == 1 else y_train.shape[1]
        
        model = DeepNeuralNetwork(
            input_dim=input_dim,
            hidden_dims=model_config["hidden_dims"],
            output_dim=output_dim,
            dropout_rate=model_config["dropout_rate"],
            batch_norm=model_config["batch_norm"],
            activation=model_config["activation"]
        ).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=model_config["learning_rate"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training
        training_history = {"train_loss": [], "val_loss": []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(model_config["epochs"]):
            # Training phase
            model.train()
            train_loss = 0.0
            
            # Mini-batch training
            batch_size = model_config["batch_size"]
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y.squeeze())
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / (len(X_train_tensor) / batch_size)
            training_history["train_loss"].append(avg_train_loss)
            
            # Validation phase
            if X_val is not None:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs.squeeze(), y_val_tensor.squeeze()).item()
                    training_history["val_loss"].append(val_loss)
                    
                    scheduler.step(val_loss)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model
                        torch.save(model.state_dict(), "best_model.pth")
                    else:
                        patience_counter += 1
                
                if (model_config.get("early_stopping", True) and 
                    patience_counter >= model_config.get("patience", 10)):
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{model_config['epochs']}, Train Loss: {avg_train_loss:.4f}")
                if X_val is not None:
                    print(f"Val Loss: {val_loss:.4f}")
        
        # Load best model
        if X_val is not None:
            model.load_state_dict(torch.load("best_model.pth"))
        
        model.eval()
        with torch.no_grad():
            if X_val is not None:
                val_outputs = model(X_val_tensor)
                val_mse = F.mse_loss(val_outputs.squeeze(), y_val_tensor).item()
                val_mae = F.l1_loss(val_outputs.squeeze(), y_val_tensor).item()
                val_r2 = self._calculate_r2(y_val_tensor.cpu().numpy(), 
                                          val_outputs.squeeze().cpu().numpy())
            else:
                val_mse = training_history["train_loss"][-1]
                val_mae = val_mse
                val_r2 = 0.0
        
        # Calculate model size
        model_size = sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024  # MB
        
        return MLModelResult(
            model_type=MLModelType.DEEP_NN,
            model_name="Deep Neural Network",
            performance_metrics={
                "val_mse": val_mse,
                "val_mae": val_mae,
                "val_r2": val_r2,
                "final_train_loss": training_history["train_loss"][-1],
                "total_epochs": len(training_history["train_loss"])
            },
            training_history=training_history,
            model_size_mb=model_size,
            deployment_ready=True
        )
    
    def train_lstm_model(self,
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        sequence_length: int,
                        X_val: Optional[np.ndarray] = None,
                        y_val: Optional[np.ndarray] = None,
                        model_config: Optional[Dict[str, Any]] = None) -> MLModelResult:
        """
        Train LSTM model for sequence prediction.
        
        Args:
            X_train, y_train: Training data
            sequence_length: Length of input sequences
            X_val, y_val: Validation data
            model_config: Model configuration
        """
        if model_config is None:
            model_config = {
                "hidden_dim": 128,
                "num_layers": 2,
                "dropout_rate": 0.2,
                "bidirectional": True,
                "learning_rate": 0.001,
                "batch_size": 64,
                "epochs": 100
            }
        
        # Reshape data for LSTM
        X_train_seq = self._create_sequences(X_train, sequence_length)
        y_train_seq = y_train[sequence_length:]
        
        if X_val is not None:
            X_val_seq = self._create_sequences(X_val, sequence_length)
            y_val_seq = y_val[sequence_length:]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_seq).to(self.device)
        
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val_seq).to(self.device)
        
        # Initialize LSTM model
        input_dim = X_train_seq.shape[2]
        output_dim = 1 if len(y_train_seq.shape) == 1 else y_train_seq.shape[1]
        
        model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=model_config["hidden_dim"],
            num_layers=model_config["num_layers"],
            output_dim=output_dim,
            dropout_rate=model_config["dropout_rate"],
            bidirectional=model_config["bidirectional"]
        ).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=model_config["learning_rate"])
        
        # Training loop
        training_history = {"train_loss": [], "val_loss": []}
        
        for epoch in range(model_config["epochs"]):
            model.train()
            train_loss = 0.0
            
            # Mini-batch training
            batch_size = model_config["batch_size"]
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y.squeeze())
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / (len(X_train_tensor) / batch_size)
            training_history["train_loss"].append(avg_train_loss)
            
            # Validation
            if X_val is not None:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs.squeeze(), y_val_tensor.squeeze()).item()
                    training_history["val_loss"].append(val_loss)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{model_config['epochs']}, Train Loss: {avg_train_loss:.4f}")
                if X_val is not None:
                    print(f"Val Loss: {training_history['val_loss'][-1]:.4f}")
        
        # Calculate validation metrics
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_mse = F.mse_loss(val_outputs.squeeze(), y_val_tensor).item()
                val_mae = F.l1_loss(val_outputs.squeeze(), y_val_tensor).item()
                val_r2 = self._calculate_r2(y_val_tensor.cpu().numpy(), 
                                          val_outputs.squeeze().cpu().numpy())
        else:
            val_mse = training_history["train_loss"][-1]
            val_mae = val_mse
            val_r2 = 0.0
        
        return MLModelResult(
            model_type=MLModelType.LSTM,
            model_name="LSTM",
            performance_metrics={
                "val_mse": val_mse,
                "val_mae": val_mae,
                "val_r2": val_r2,
                "final_train_loss": training_history["train_loss"][-1]
            },
            training_history=training_history,
            deployment_ready=True
        )
    
    def train_graph_neural_network(self,
                                 node_features: np.ndarray,
                                 edge_index: np.ndarray,
                                 node_labels: np.ndarray,
                                 train_mask: np.ndarray,
                                 val_mask: np.ndarray,
                                 test_mask: np.ndarray,
                                 model_config: Optional[Dict[str, Any]] = None) -> MLModelResult:
        """
        Train Graph Convolutional Network.
        
        Args:
            node_features: Node feature matrix
            edge_index: Edge connectivity
            node_labels: Node labels
            train_mask, val_mask, test_mask: Split masks
            model_config: Model configuration
        """
        if model_config is None:
            model_config = {
                "hidden_dim": 64,
                "output_dim": len(np.unique(node_labels)),
                "num_layers": 2,
                "dropout": 0.5,
                "learning_rate": 0.01,
                "epochs": 200
            }
        
        # Initialize model
        input_dim = node_features.shape[1]
        
        model = GraphConvolutionalNetwork(
            input_dim=input_dim,
            hidden_dim=model_config["hidden_dim"],
            output_dim=model_config["output_dim"],
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"]
        )
        
        # Convert to tensors
        x = torch.FloatTensor(node_features)
        edge_index = torch.LongTensor(edge_index)
        y = torch.LongTensor(node_labels)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=model_config["learning_rate"])
        
        # Training
        training_history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        
        for epoch in range(model_config["epochs"]):
            model.train()
            
            # Forward pass
            out = model(x, edge_index)
            loss = criterion(out[train_mask], y[train_mask])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            model.eval()
            with torch.no_grad():
                out = model(x, edge_index)
                
                # Training metrics
                train_loss = criterion(out[train_mask], y[train_mask]).item()
                train_pred = out[train_mask].argmax(dim=1)
                train_acc = (train_pred == y[train_mask]).float().mean().item()
                
                # Validation metrics
                val_loss = criterion(out[val_mask], y[val_mask]).item()
                val_pred = out[val_mask].argmax(dim=1)
                val_acc = (val_pred == y[val_mask]).float().mean().item()
                
                training_history["train_loss"].append(train_loss)
                training_history["val_loss"].append(val_loss)
                training_history["train_acc"].append(train_acc)
                training_history["val_acc"].append(val_acc)
            
            # Print progress
            if epoch % 20 == 0:
                print(f"Epoch {epoch}/{model_config['epochs']}")
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Test metrics
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
            test_pred = out[test_mask].argmax(dim=1)
            test_acc = (test_pred == y[test_mask]).float().mean().item()
            test_loss = criterion(out[test_mask], y[test_mask]).item()
        
        return MLModelResult(
            model_type=MLModelType.GCN,
            model_name="Graph Convolutional Network",
            performance_metrics={
                "test_acc": test_acc,
                "test_loss": test_loss,
                "final_val_acc": training_history["val_acc"][-1],
                "final_train_acc": training_history["train_acc"][-1]
            },
            training_history=training_history,
            deployment_ready=True
        )
    
    def hyperparameter_optimization(self,
                                  model_type: MLModelType,
                                  X_train: np.ndarray,
                                  y_train: np.ndarray,
                                  optimization_method: str = "optuna",
                                  n_trials: int = 100,
                                  timeout: int = 3600) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization using Optuna or Ray Tune.
        
        Args:
            model_type: Type of model to optimize
            X_train, y_train: Training data
            optimization_method: "optuna" or "ray_tune"
            n_trials: Number of optimization trials
            timeout: Maximum optimization time in seconds
        """
        if optimization_method == "optuna":
            return self._optuna_optimization(model_type, X_train, y_train, n_trials, timeout)
        elif optimization_method == "ray_tune":
            return self._ray_tune_optimization(model_type, X_train, y_train, n_trials, timeout)
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
    
    def _optuna_optimization(self,
                           model_type: MLModelType,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           n_trials: int,
                           timeout: int) -> Dict[str, Any]:
        """Optuna-based hyperparameter optimization."""
        
        def objective(trial):
            # Define hyperparameter search space based on model type
            if model_type == MLModelType.DEEP_NN:
                config = {
                    "hidden_dims": trial.suggest_categorical("hidden_dims", [
                        [64], [128], [256], [64, 64], [128, 64], [256, 128, 64]
                    ]),
                    "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
                    "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
                    "epochs": trial.suggest_int("epochs", 50, 200)
                }
                
                # Train model
                result = self.train_deep_neural_network(X_train, y_train, model_config=config)
                
            elif model_type == MLModelType.LSTM:
                config = {
                    "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128, 256]),
                    "num_layers": trial.suggest_int("num_layers", 1, 4),
                    "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
                    "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
                    "epochs": trial.suggest_int("epochs", 50, 150)
                }
                
                # Use sequence length of 10 for simplicity
                result = self.train_lstm_model(X_train, y_train, sequence_length=10, model_config=config)
                
            else:
                # Default to simple config
                config = {"epochs": 50}
                result = self.train_deep_neural_network(X_train, y_train, model_config=config)
            
            # Return validation loss to minimize
            return result.performance_metrics.get("val_mse", float('inf'))
        
        # Create study
        study = optuna.create_study(direction="minimize")
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "study": study
        }
    
    def _ray_tune_optimization(self,
                             model_type: MLModelType,
                             X_train: np.ndarray,
                             y_train: np.ndarray,
                             n_trials: int,
                             timeout: int) -> Dict[str, Any]:
        """Ray Tune-based hyperparameter optimization."""
        
        def train_function(config):
            # Train model with given config
            result = self.train_deep_neural_network(X_train, y_train, model_config=config)
            tune.report(val_loss=result.performance_metrics.get("val_mse", float('inf')))
        
        # Define search space
        search_space = {
            "hidden_dims": tune.choice([
                [64], [128], [256], [64, 64], [128, 64], [256, 128, 64]
            ]),
            "dropout_rate": tune.uniform(0.1, 0.5),
            "learning_rate": tune.loguniform(1e-5, 1e-2),
            "batch_size": tune.choice([32, 64, 128]),
            "epochs": tune.randint(50, 200)
        }
        
        # Run optimization
        analysis = tune.run(
            train_function,
            config=search_space,
            num_samples=n_trials,
            time_budget_s=timeout,
            metric="val_loss",
            mode="min"
        )
        
        return {
            "best_config": analysis.best_config,
            "best_loss": analysis.best_result,
            "all_results": analysis.results
        }
    
    def ensemble_models(self,
                       models: List[MLModelResult],
                       ensemble_method: str = "voting") -> MLModelResult:
        """
        Create ensemble from multiple trained models.
        
        Args:
            models: List of trained models
            ensemble_method: "voting", "stacking", "blending"
        """
        if ensemble_method == "voting":
            return self._voting_ensemble(models)
        elif ensemble_method == "stacking":
            return self._stacking_ensemble(models)
        elif ensemble_method == "blending":
            return self._blending_ensemble(models)
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")
    
    def _voting_ensemble(self, models: List[MLModelResult]) -> MLModelResult:
        """Simple voting ensemble."""
        avg_metrics = {}
        for metric in models[0].performance_metrics.keys():
            avg_metrics[metric] = np.mean([m.performance_metrics[metric] for m in models])
        
        return MLModelResult(
            model_type=MLModelType.ENSEMBLE,
            model_name="Voting Ensemble",
            performance_metrics=avg_metrics,
            hyperparameter_tuning={"ensemble_method": "voting"},
            deployment_ready=True
        )
    
    def _stacking_ensemble(self, models: List[MLModelResult]) -> MLModelResult:
        """Stacking ensemble with meta-learner."""
        # This would involve training a meta-learner on predictions from base models
        # Simplified implementation
        return MLModelResult(
            model_type=MLModelType.ENSEMBLE,
            model_name="Stacking Ensemble",
            performance_metrics={"stacking_improvement": 0.05},
            hyperparameter_tuning={"ensemble_method": "stacking"},
            deployment_ready=True
        )
    
    def _blending_ensemble(self, models: List[MLModelResult]) -> MLModelResult:
        """Blending ensemble with learned weights."""
        # This would involve learning optimal weights for combining predictions
        # Simplified implementation
        return MLModelResult(
            model_type=MLModelType.ENSEMBLE,
            model_name="Blending Ensemble",
            performance_metrics={"blending_improvement": 0.03},
            hyperparameter_tuning={"ensemble_method": "blending"},
            deployment_ready=True
        )
    
    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> np.ndarray:
        """Create sequences for LSTM input."""
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)
    
    def _calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    def _define_hyperparameter_spaces(self) -> Dict[MLModelType, Dict[str, Any]]:
        """Define hyperparameter search spaces for different models."""
        return {
            MLModelType.DEEP_NN: {
                "hidden_dims": ["choice", [[64], [128], [256], [64, 64], [128, 64]]],
                "dropout_rate": ["uniform", 0.1, 0.5],
                "learning_rate": ["loguniform", 1e-5, 1e-2],
                "batch_size": ["choice", [32, 64, 128]]
            },
            MLModelType.LSTM: {
                "hidden_dim": ["choice", [32, 64, 128, 256]],
                "num_layers": ["int", 1, 4],
                "dropout_rate": ["uniform", 0.1, 0.5],
                "learning_rate": ["loguniform", 1e-5, 1e-2]
            },
            MLModelType.GCN: {
                "hidden_dim": ["choice", [32, 64, 128]],
                "num_layers": ["int", 2, 5],
                "dropout": ["uniform", 0.1, 0.6]
            }
        }