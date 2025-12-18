"""
Autoencoder Models for Anomaly Detection and Dimensionality Reduction
Provides advanced autoencoder architectures with multiple use cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class AutoencoderResult:
    """Result of autoencoder training."""
    reconstruction_error: np.ndarray
    reconstruction_threshold: float
    anomalies: np.ndarray
    anomaly_scores: np.ndarray
    latent_representation: np.ndarray
    model_weights: Dict[str, np.ndarray]
    training_history: Dict[str, List[float]]
    anomaly_metrics: Dict[str, float]


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder for generative modeling and anomaly detection."""
    
    def __init__(self, 
                 input_dim: int,
                 latent_dim: int,
                 hidden_dims: List[int] = None,
                 activation: str = "relu"):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
            "swish": nn.SiLU(),
            "gelu": nn.GELU()
        }
        return activations.get(activation, nn.ReLU())
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


class DenoisingAutoencoder(nn.Module):
    """Denoising Autoencoder for robust feature learning."""
    
    def __init__(self, 
                 input_dim: int,
                 latent_dim: int,
                 hidden_dims: List[int] = None,
                 activation: str = "relu",
                 noise_level: float = 0.1):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        self.noise_level = noise_level
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self._get_activation(activation)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.latent = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self._get_activation(activation)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
            "swish": nn.SiLU()
        }
        return activations.get(activation, nn.ReLU())
    
    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to input."""
        noise = torch.randn_like(x) * self.noise_level
        return x + noise
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        noisy_x = self.add_noise(x)
        return self.latent(self.encoder(noisy_x))
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        z = self.encode(x)
        recon_x = self.decode(z)
        return recon_x, z


class ConvolutionalAutoencoder(nn.Module):
    """Convolutional Autoencoder for image and sequence data."""
    
    def __init__(self, 
                 input_channels: int,
                 latent_dim: int,
                 activation: str = "relu"):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            self._get_activation(activation),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            self._get_activation(activation),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            self._get_activation(activation),
            
            # Flatten and compress to latent
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, latent_dim),
            nn.Tanh()  # Bound latent space
        )
        
        # Decoder
        self.decoder_dense = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            self._get_activation(activation)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            self._get_activation(activation),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            self._get_activation(activation),
            
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # For normalized inputs
        )
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU()
        }
        return activations.get(activation, nn.ReLU())
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        # Reshape from latent
        x = self.decoder_dense(z)
        x = x.view(x.size(0), 128, 8, 8)
        return self.decoder(x)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        z = self.encode(x)
        recon_x = self.decode(z)
        return recon_x, z


class LSTMAutoencoder(nn.Module):
    """LSTM-based Autoencoder for sequential data."""
    
    def __init__(self, 
                 input_dim: int,
                 latent_dim: int,
                 hidden_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Latent space compression
        encoder_output_dim = hidden_dim * 2  # bidirectional
        self.latent_compress = nn.Linear(encoder_output_dim, latent_dim)
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            1,  # Input will be compressed features
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim * 2, input_dim)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence to latent representation."""
        # LSTM encoding
        encoder_output, (hidden, cell) = self.encoder_lstm(x)
        
        # Use the final hidden state
        hidden = hidden.view(hidden.size(0), hidden.size(1), -1)  # Flatten layers
        final_hidden = hidden[:, -1, :]  # Take last layer
        
        # Compress to latent space
        latent = self.latent_compress(final_hidden)
        return latent
    
    def decode(self, z: torch.Tensor, target_length: int) -> torch.Tensor:
        """Decode from latent representation to sequence."""
        batch_size = z.size(0)
        
        # Prepare decoder input
        decoder_input = z.unsqueeze(1).repeat(1, target_length, 1)  # Repeat across time
        
        # LSTM decoding
        decoder_output, _ = self.decoder_lstm(decoder_input)
        
        # Project to output dimension
        output = self.output_projection(decoder_output)
        return output
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        batch_size, seq_len, input_dim = x.shape
        
        # Encode
        z = self.encode(x)
        
        # Decode
        recon_x = self.decode(z, seq_len)
        
        return recon_x, z


class AutoencoderModel:
    """
    Advanced Autoencoder for anomaly detection, dimensionality reduction,
    and feature learning with multiple architectures.
    """
    
    def __init__(self, device: str = "auto"):
        """Initialize autoencoder."""
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else device)
        self.models = {}
        self.training_history = {}
        
    def train_variational_autoencoder(self,
                                    data: np.ndarray,
                                    latent_dim: int = 20,
                                    batch_size: int = 64,
                                    epochs: int = 100,
                                    learning_rate: float = 0.001,
                                    beta: float = 1.0) -> AutoencoderResult:
        """
        Train Variational Autoencoder.
        
        Args:
            data: Input data (samples x features)
            latent_dim: Dimension of latent space
            batch_size: Training batch size
            epochs: Number of training epochs
            learning_rate: Learning rate
            beta: Beta parameter for KL divergence weighting
        """
        input_dim = data.shape[1]
        
        # Initialize model
        model = VariationalAutoencoder(input_dim, latent_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Data preparation
        data_tensor = torch.FloatTensor(data).to(self.device)
        dataset = TensorDataset(data_tensor, data_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training
        history = {"reconstruction_loss": [], "kl_loss": [], "total_loss": []}
        
        for epoch in range(epochs):
            model.train()
            total_recon_loss = 0
            total_kl_loss = 0
            
            for batch_data, _ in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                recon_batch, mu, logvar = model(batch_data)
                
                # Compute losses
                recon_loss = F.mse_loss(recon_batch, batch_data, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                
                # Total loss
                total_loss = recon_loss + beta * kl_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
            
            # Record epoch metrics
            avg_recon_loss = total_recon_loss / len(data)
            avg_kl_loss = total_kl_loss / len(data)
            avg_total_loss = avg_recon_loss + beta * avg_kl_loss
            
            history["reconstruction_loss"].append(avg_recon_loss)
            history["kl_loss"].append(avg_kl_loss)
            history["total_loss"].append(avg_total_loss)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}/{epochs}, Recon Loss: {avg_recon_loss:.4f}, "
                      f"KL Loss: {avg_kl_loss:.4f}, Total: {avg_total_loss:.4f}")
        
        # Evaluation on full dataset
        model.eval()
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data).to(self.device)
            recon_data, mu, logvar = model(data_tensor)
            
            # Calculate reconstruction errors
            reconstruction_errors = F.mse_loss(recon_data, data_tensor, reduction='none').mean(dim=1).cpu().numpy()
            
            # Set threshold for anomaly detection
            threshold = np.percentile(reconstruction_errors, 95)  # Top 5% as anomalies
            
            # Identify anomalies
            anomalies = reconstruction_errors > threshold
            anomaly_scores = reconstruction_errors
            
            # Get latent representations
            latent_representations = mu.cpu().numpy()
        
        # Calculate metrics
        if np.sum(anomalies) > 0:
            anomaly_rate = np.mean(anomalies)
        else:
            anomaly_rate = 0
        
        result = AutoencoderResult(
            reconstruction_error=reconstruction_errors,
            reconstruction_threshold=threshold,
            anomalies=anomalies,
            anomaly_scores=anomaly_scores,
            latent_representation=latent_representations,
            model_weights={name: param.data.cpu().numpy() for name, param in model.named_parameters()},
            training_history=history,
            anomaly_metrics={
                "anomaly_rate": anomaly_rate,
                "threshold": threshold,
                "mean_reconstruction_error": np.mean(reconstruction_errors),
                "std_reconstruction_error": np.std(reconstruction_errors)
            }
        )
        
        self.models["vae"] = model
        return result
    
    def train_denoising_autoencoder(self,
                                  data: np.ndarray,
                                  latent_dim: int = 10,
                                  noise_level: float = 0.1,
                                  batch_size: int = 64,
                                  epochs: int = 100,
                                  learning_rate: float = 0.001) -> AutoencoderResult:
        """
        Train Denoising Autoencoder.
        
        Args:
            data: Clean input data
            latent_dim: Dimension of latent space
            noise_level: Level of noise to add during training
            batch_size: Training batch size
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        input_dim = data.shape[1]
        
        # Initialize model
        model = DenoisingAutoencoder(input_dim, latent_dim, noise_level=noise_level).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Data preparation
        data_tensor = torch.FloatTensor(data).to(self.device)
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training
        history = {"reconstruction_loss": []}
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch_data in dataloader:
                batch_data = batch_data[0].to(self.device)
                optimizer.zero_grad()
                
                # Forward pass
                recon_batch, _ = model(batch_data)
                
                # Compute reconstruction loss (denoising)
                loss = F.mse_loss(recon_batch, batch_data, reduction='sum')
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Record epoch metrics
            avg_loss = total_loss / len(data)
            history["reconstruction_loss"].append(avg_loss)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data).to(self.device)
            recon_data, latent = model(data_tensor)
            
            # Calculate reconstruction errors
            reconstruction_errors = F.mse_loss(recon_data, data_tensor, reduction='none').mean(dim=1).cpu().numpy()
            
            # Set threshold for anomaly detection
            threshold = np.percentile(reconstruction_errors, 95)
            
            # Identify anomalies
            anomalies = reconstruction_errors > threshold
            anomaly_scores = reconstruction_errors
            
            # Get latent representations
            latent_representations = latent.cpu().numpy()
        
        result = AutoencoderResult(
            reconstruction_error=reconstruction_errors,
            reconstruction_threshold=threshold,
            anomalies=anomalies,
            anomaly_scores=anomaly_scores,
            latent_representation=latent_representations,
            model_weights={name: param.data.cpu().numpy() for name, param in model.named_parameters()},
            training_history=history,
            anomaly_metrics={
                "anomaly_rate": np.mean(anomalies),
                "noise_level": noise_level,
                "latent_dim": latent_dim
            }
        )
        
        self.models["denoising_ae"] = model
        return result
    
    def train_convolutional_autoencoder(self,
                                      data: np.ndarray,
                                      input_channels: int,
                                      image_height: int,
                                      image_width: int,
                                      latent_dim: int = 64,
                                      batch_size: int = 32,
                                      epochs: int = 50,
                                      learning_rate: float = 0.001) -> AutoencoderResult:
        """
        Train Convolutional Autoencoder for image data.
        
        Args:
            data: Input images (samples x channels x height x width)
            input_channels: Number of input channels
            image_height: Height of images
            image_width: Width of images
            latent_dim: Dimension of latent space
            batch_size: Training batch size
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        # Reshape data to proper format
        if len(data.shape) == 3:
            data = data.reshape(-1, input_channels, image_height, image_width)
        
        # Initialize model
        model = ConvolutionalAutoencoder(input_channels, latent_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Data preparation
        data_tensor = torch.FloatTensor(data).to(self.device)
        dataset = TensorDataset(data_tensor, data_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training
        history = {"reconstruction_loss": []}
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch_data, _ in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                recon_batch, _ = model(batch_data)
                
                # Compute reconstruction loss
                loss = F.mse_loss(recon_batch, batch_data, reduction='sum')
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Record epoch metrics
            avg_loss = total_loss / len(data)
            history["reconstruction_loss"].append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            recon_data, latent = model(data_tensor)
            
            # Calculate reconstruction errors
            reconstruction_errors = F.mse_loss(recon_data, data_tensor, reduction='none').mean(dim=(1, 2, 3)).cpu().numpy()
            
            # Set threshold for anomaly detection
            threshold = np.percentile(reconstruction_errors, 95)
            
            # Identify anomalies
            anomalies = reconstruction_errors > threshold
            anomaly_scores = reconstruction_errors
            
            # Get latent representations
            latent_representations = latent.cpu().numpy()
        
        result = AutoencoderResult(
            reconstruction_error=reconstruction_errors,
            reconstruction_threshold=threshold,
            anomalies=anomalies,
            anomaly_scores=anomaly_scores,
            latent_representation=latent_representations,
            model_weights={name: param.data.cpu().numpy() for name, param in model.named_parameters()},
            training_history=history,
            anomaly_metrics={
                "anomaly_rate": np.mean(anomalies),
                "latent_dim": latent_dim,
                "image_shape": (input_channels, image_height, image_width)
            }
        )
        
        self.models["conv_ae"] = model
        return result
    
    def train_lstm_autoencoder(self,
                             data: np.ndarray,
                             sequence_length: int,
                             latent_dim: int = 32,
                             hidden_dim: int = 64,
                             batch_size: int = 32,
                             epochs: int = 100,
                             learning_rate: float = 0.001) -> AutoencoderResult:
        """
        Train LSTM Autoencoder for sequential data.
        
        Args:
            data: Sequential data
            sequence_length: Length of input sequences
            latent_dim: Dimension of latent space
            hidden_dim: LSTM hidden dimension
            batch_size: Training batch size
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        input_dim = data.shape[1] if len(data.shape) > 1 else 1
        
        # Create sequences
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        
        if len(sequences) == 0:
            raise ValueError("Data too short for specified sequence length")
        
        sequences = np.array(sequences)
        
        # Initialize model
        model = LSTMAutoencoder(input_dim, latent_dim, hidden_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Data preparation
        data_tensor = torch.FloatTensor(sequences).to(self.device)
        dataset = TensorDataset(data_tensor, data_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training
        history = {"reconstruction_loss": []}
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch_data, _ in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                recon_batch, _ = model(batch_data)
                
                # Compute reconstruction loss
                loss = F.mse_loss(recon_batch, batch_data, reduction='sum')
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Record epoch metrics
            avg_loss = total_loss / len(sequences)
            history["reconstruction_loss"].append(avg_loss)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            recon_data, latent = model(data_tensor)
            
            # Calculate reconstruction errors
            reconstruction_errors = F.mse_loss(recon_data, data_tensor, reduction='none').mean(dim=(1, 2)).cpu().numpy()
            
            # Set threshold for anomaly detection
            threshold = np.percentile(reconstruction_errors, 95)
            
            # Identify anomalies
            anomalies = reconstruction_errors > threshold
            anomaly_scores = reconstruction_errors
            
            # Get latent representations
            latent_representations = latent.cpu().numpy()
        
        result = AutoencoderResult(
            reconstruction_error=reconstruction_errors,
            reconstruction_threshold=threshold,
            anomalies=anomalies,
            anomaly_scores=anomaly_scores,
            latent_representation=latent_representations,
            model_weights={name: param.data.cpu().numpy() for name, param in model.named_parameters()},
            training_history=history,
            anomaly_metrics={
                "anomaly_rate": np.mean(anomalies),
                "sequence_length": sequence_length,
                "latent_dim": latent_dim
            }
        )
        
        self.models["lstm_ae"] = model
        return result
    
    def detect_anomalies(self, 
                       data: np.ndarray,
                       model_name: str,
                       threshold_percentile: float = 95) -> Dict[str, Any]:
        """
        Detect anomalies using trained autoencoder.
        
        Args:
            data: Input data for anomaly detection
            model_name: Name of trained model
            threshold_percentile: Percentile threshold for anomalies
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        model.eval()
        
        # Prepare data
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data).to(self.device)
            
            if model_name == "vae":
                recon_data, mu, logvar = model(data_tensor)
                reconstruction_errors = F.mse_loss(recon_data, data_tensor, reduction='none').mean(dim=1).cpu().numpy()
            elif model_name == "denoising_ae":
                recon_data, latent = model(data_tensor)
                reconstruction_errors = F.mse_loss(recon_data, data_tensor, reduction='none').mean(dim=1).cpu().numpy()
            elif model_name == "conv_ae":
                if len(data_tensor.shape) == 2:
                    # Assume flattened images
                    batch_size = data_tensor.size(0)
                    img_size = int(np.sqrt(data_tensor.size(1) // model.encoder[0].in_channels))
                    data_tensor = data_tensor.view(batch_size, model.encoder[0].in_channels, img_size, img_size)
                
                recon_data, latent = model(data_tensor)
                reconstruction_errors = F.mse_loss(recon_data, data_tensor, reduction='none').mean(dim=(1, 2, 3)).cpu().numpy()
            elif model_name == "lstm_ae":
                recon_data, latent = model(data_tensor)
                reconstruction_errors = F.mse_loss(recon_data, data_tensor, reduction='none').mean(dim=(1, 2)).cpu().numpy()
            else:
                raise ValueError(f"Unknown model type: {model_name}")
        
        # Set threshold and detect anomalies
        threshold = np.percentile(reconstruction_errors, threshold_percentile)
        anomalies = reconstruction_errors > threshold
        anomaly_scores = reconstruction_errors / threshold  # Normalized scores
        
        return {
            "anomalies": anomalies,
            "anomaly_scores": anomaly_scores,
            "reconstruction_errors": reconstruction_errors,
            "threshold": threshold,
            "anomaly_indices": np.where(anomalies)[0].tolist(),
            "anomaly_rate": np.mean(anomalies)
        }
    
    def generate_samples(self, 
                        model_name: str,
                        num_samples: int = 10,
                        latent_sampling: str = "random") -> np.ndarray:
        """
        Generate new samples using trained autoencoder.
        
        Args:
            model_name: Name of trained model
            num_samples: Number of samples to generate
            latent_sampling: "random" or "from_data"
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        model.eval()
        
        if model_name == "vae":
            return self._generate_vae_samples(model, num_samples, latent_sampling)
        else:
            raise NotImplementedError(f"Sample generation not implemented for {model_name}")
    
    def _generate_vae_samples(self, 
                            model: VariationalAutoencoder,
                            num_samples: int,
                            sampling_method: str) -> np.ndarray:
        """Generate samples using VAE."""
        with torch.no_grad():
            if sampling_method == "random":
                # Sample from standard normal distribution
                z = torch.randn(num_samples, model.latent_dim).to(self.device)
            else:
                # This would require storing training data latent representations
                # For simplicity, use random sampling
                z = torch.randn(num_samples, model.latent_dim).to(self.device)
            
            # Decode samples
            generated = model.decode(z)
            
            return generated.cpu().numpy()
    
    def visualize_latent_space(self,
                             data: np.ndarray,
                             model_name: str,
                             labels: Optional[np.ndarray] = None,
                             save_path: Optional[str] = None) -> None:
        """
        Visualize latent space representations.
        
        Args:
            data: Input data
            model_name: Name of trained model
            labels: Optional labels for coloring points
            save_path: Path to save the visualization
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        model.eval()
        
        # Get latent representations
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data).to(self.device)
            
            if model_name == "vae":
                _, mu, _ = model.encode(data_tensor)
                latent_reps = mu.cpu().numpy()
            else:
                latent_reps = model.encode(data_tensor).cpu().numpy()
        
        # If latent space is high-dimensional, reduce with PCA
        if latent_reps.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            latent_reps = pca.fit_transform(latent_reps)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(latent_reps[mask, 0], latent_reps[mask, 1], 
                           c=[colors[i]], label=f'Class {label}', alpha=0.7)
            plt.legend()
        else:
            plt.scatter(latent_reps[:, 0], latent_reps[:, 1], alpha=0.7)
        
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title(f'Latent Space Visualization - {model_name}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class AnomalyAutoencoder:
    """
    Specialized Autoencoder for anomaly detection with domain-specific optimizations.
    """
    
    def __init__(self, device: str = "auto"):
        """Initialize anomaly autoencoder."""
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else device)
        self.threshold_model = None
        self.preprocessing_pipeline = None
        
    def train_domain_specific_autoencoder(self,
                                        data: pd.DataFrame,
                                        domain: str = "financial",
                                        anomaly_ratio: float = 0.05,
                                        autoencoder_type: str = "variational") -> AutoencoderResult:
        """
        Train domain-specific autoencoder for anomaly detection.
        
        Args:
            data: Input data
            domain: Domain type ("financial", "industrial", "cybersecurity", "healthcare")
            anomaly_ratio: Expected ratio of anomalies
            autoencoder_type: Type of autoencoder to use
        """
        # Preprocess data based on domain
        processed_data = self._preprocess_by_domain(data, domain)
        
        # Initialize autoencoder
        ae_model = AutoencoderModel(device=self.device)
        
        # Train based on domain and autoencoder type
        if autoencoder_type == "variational":
            return ae_model.train_variational_autoencoder(
                processed_data, 
                latent_dim=min(20, processed_data.shape[1] // 4)
            )
        elif autoencoder_type == "denoising":
            return ae_model.train_denoising_autoencoder(
                processed_data,
                latent_dim=min(10, processed_data.shape[1] // 6)
            )
        else:
            raise ValueError(f"Unknown autoencoder type: {autoencoder_type}")
    
    def _preprocess_by_domain(self, data: pd.DataFrame, domain: str) -> np.ndarray:
        """Preprocess data based on domain."""
        # Handle categorical variables
        data_processed = data.copy()
        
        # Remove or encode categorical variables
        categorical_cols = data_processed.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            # Simple one-hot encoding
            data_processed = pd.get_dummies(data_processed, columns=categorical_cols, prefix=categorical_cols)
        
        # Domain-specific preprocessing
        if domain == "financial":
            # Handle missing values and outliers
            data_processed = data_processed.fillna(data_processed.median())
            
            # Log transform for positive financial metrics
            positive_cols = data_processed.select_dtypes(include=[np.number]).columns
            for col in positive_cols:
                if (data_processed[col] > 0).all():
                    data_processed[f"{col}_log"] = np.log1p(data_processed[col])
        
        elif domain == "industrial":
            # Normalize time-series data
            numeric_cols = data_processed.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                data_processed[f"{col}_norm"] = (data_processed[col] - data_processed[col].mean()) / data_processed[col].std()
        
        elif domain == "cybersecurity":
            # Binary encoding for security events
            data_processed = (data_processed > data_processed.median()).astype(int)
        
        elif domain == "healthcare":
            # Preserve structure for medical data
            data_processed = data_processed.fillna(0)  # Medical data often has meaningful zeros
        
        return data_processed.values
    
    def adaptive_threshold_learning(self, 
                                  reconstruction_errors: np.ndarray,
                                  known_anomalies: Optional[np.ndarray] = None) -> float:
        """
        Learn optimal anomaly threshold using domain knowledge or historical anomalies.
        
        Args:
            reconstruction_errors: Reconstruction errors from autoencoder
            known_anomalies: Known anomaly labels for supervised threshold tuning
        """
        if known_anomalies is not None:
            # Supervised threshold learning
            from sklearn.metrics import precision_recall_curve
            
            # Find optimal threshold using precision-recall
            precisions, recalls, thresholds = precision_recall_curve(known_anomalies, reconstruction_errors)
            
            # F1 score for each threshold
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else np.percentile(reconstruction_errors, 95)
            
        else:
            # Unsupervised threshold using percentile
            optimal_threshold = np.percentile(reconstruction_errors, 95)
        
        self.threshold_model = optimal_threshold
        return optimal_threshold
    
    def ensemble_autoencoder_anomaly_detection(self,
                                             data: np.ndarray,
                                             autoencoder_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ensemble multiple autoencoder architectures for robust anomaly detection.
        
        Args:
            data: Input data
            autoencoder_configs: List of autoencoder configurations
        """
        results = []
        
        # Train multiple autoencoders
        ae_model = AutoencoderModel(device=self.device)
        
        for config in autoencoder_configs:
            ae_type = config.get("type", "variational")
            
            if ae_type == "variational":
                result = ae_model.train_variational_autoencoder(
                    data,
                    latent_dim=config.get("latent_dim", 20)
                )
            elif ae_type == "denoising":
                result = ae_model.train_denoising_autoencoder(
                    data,
                    latent_dim=config.get("latent_dim", 10),
                    noise_level=config.get("noise_level", 0.1)
                )
            else:
                continue
            
            results.append(result)
        
        # Combine anomaly scores
        all_scores = np.array([r.anomaly_scores for r in results])
        ensemble_scores = np.mean(all_scores, axis=0)
        
        # Determine ensemble threshold
        ensemble_threshold = np.percentile(ensemble_scores, 95)
        ensemble_anomalies = ensemble_scores > ensemble_threshold
        
        return {
            "ensemble_scores": ensemble_scores,
            "ensemble_anomalies": ensemble_anomalies,
            "ensemble_threshold": ensemble_threshold,
            "individual_results": results,
            "anomaly_rate": np.mean(ensemble_anomalies),
            "score_std": np.std(all_scores, axis=0),
            "confidence": 1 - (np.std(all_scores, axis=0) / (ensemble_scores + 1e-10))
        }