import torch.nn.functional as F
import torch

class CrossEntropyFromLogits:
    """
    Implements the combination of:
    - Softmax activation (from raw logits)
    - Cross-entropy loss

    This is a common choice for multi-class classification.
    """

    def forward(self, Z, Y):
        """
        Forward pass: compute the cross-entropy loss from raw logits.

        Args:
            Z (torch.Tensor): Logits (batch_size, n_classes).
            Y (torch.Tensor): True labels (batch_size,).
        """
        self.Y = Y
        # Softmax probabilities
        self.A = F.softmax(Z, dim=1)

        # Log-softmax (más estable que log(softmax))
        log_softmax_Z = F.log_softmax(Z, dim=1)

        # Seleccionar log-probs de las clases correctas
        log_probs = log_softmax_Z[torch.arange(Z.size(0)), Y]

        # Cross-entropy loss
        loss = -log_probs.mean()
        return loss

    def backward(self, n_classes):
        """
        Backward pass: compute the gradient of the loss w.r.t. logits Z.

        Args:
            n_classes (int): Number of classes.
        """
        batch_size = self.Y.size(0)

        # One-hot encoding
        Y_one_hot = F.one_hot(self.Y, num_classes=n_classes).float()

        # Gradiente: (softmax - one_hot) / batch_size
        dZ = (self.A - Y_one_hot) / batch_size
        return dZ
