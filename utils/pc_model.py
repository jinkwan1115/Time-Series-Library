import torch
import torch.nn as nn
import numpy as np
import time
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceImage, BettiCurve

class Model(nn.Module):
    def __init__(self):
        """
        Model class to compute persistent homology for multivariate time series data.
        """
        super(Model, self).__init__()
        # Initialize the Vietoris-Rips persistence object
        self.vr_persistence = VietorisRipsPersistence(homology_dimensions=[0])

    def forward(self, x: torch.Tensor):
        """
        Forward pass to compute persistent homology.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, seq_len, num_channels].

        Returns:
            torch.Tensor: Output tensor of shape [batch, seq_len, 2], where the last dimension
                          contains birth and death pairs with [0, 0] padding if needed.
        """
        batch_size, seq_len, num_channels = x.shape

        # Prepare output storage
        persistence_diagrams = []

        # Loop through each batch
        for b in range(batch_size):
            # Extract the point cloud for the current batch
            point_cloud = x[b].cpu().numpy()  # Shape: [seq_len, num_channels]

            # Compute persistent homology
            diagrams = self.vr_persistence.fit_transform([point_cloud])

            # Extract the 0-dimensional persistence diagram (birth-death pairs)
            h0_diagrams = diagrams[0][:,:2]  # Only H0 is needed

            # Store the result
            persistence_diagrams.append(torch.tensor(h0_diagrams, dtype=torch.float32, device=x.device))

        # Pad or truncate to match seq_len
        persistence_tensor = self._adjust_diagram(persistence_diagrams, batch_size, seq_len, x.device)
        # Reverse the order of the last dimension
        persistence_tensor = persistence_tensor[..., [1, 0]]

        return persistence_tensor

    def _adjust_diagram(self, diagrams, batch_size, seq_len, device):
        """
        Adjust the size of persistence diagrams to match seq_len by padding with [0, 0] if needed.

        Args:
            diagrams (list[torch.Tensor]): List of tensors with shape [num_barcodes, 2].
            batch_size (int): Batch size.
            seq_len (int): Target sequence length.
            device (torch.device): Device to store the tensor.

        Returns:
            torch.Tensor: Adjusted tensor of shape [batch_size, seq_len, 2].
        """
        adjusted_diagrams = torch.zeros((batch_size, seq_len, 2), device=device, dtype=torch.float32)  # Fill with [0, 0]

        # Adjust each batch
        for b, diagram in enumerate(diagrams):
            num_barcodes = diagram.size(0)

            if num_barcodes > seq_len:
                adjusted_diagrams[b] = diagram[:seq_len]  # Truncate if too many barcodes
            else:
                adjusted_diagrams[b, :num_barcodes] = diagram  # Pad if not enough barcodes

        return adjusted_diagrams


class PC_Model(nn.Module):
    def __init__(self,args):
        """
        Model class to compute persistent homology for multivariate time series data.
        """
        super(PC_Model, self).__init__()
        # Initialize the Vietoris-Rips persistence object
        self.vr_persistence = VietorisRipsPersistence(homology_dimensions=[0])
        #self.resolution = args.pred_len
        #self.bc = BettiCurve(n_bins = self.resolution) #100: resolution

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, num_channels = x.shape

        # Prepare output storage
        output = torch.zeros((batch_size, seq_len), device=x.device)
        bc = BettiCurve(n_bins = seq_len)
        # Loop through each batch
        for b in range(batch_size):
            # Extract the point cloud for the current batch
            point_cloud = x[b].cpu().numpy()  # Shape: [seq_len, num_channels]

            # Compute persistent homology
            diagrams = self.vr_persistence.fit_transform([point_cloud])
            betti_curves = bc.fit_transform(diagrams)
            betti_curves = (betti_curves - betti_curves.min(axis=2, keepdims=True)) / (
                betti_curves.max(axis=2, keepdims=True) - betti_curves.min(axis=2, keepdims=True) + 1e-8
            )

            # Store the result
            output[b] = torch.tensor(betti_curves, device=x.device)

        return output