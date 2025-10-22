import torch
class ReLU:
    """
    ReLU activation layer.
    """

    def forward(self, Z):
        """
        Perform the forward pass of the ReLU activation function.

        Args:
            Z (torch.Tensor): Input tensor.

        Returns:
            A torch.Tensor: Output tensor with ReLU applied element-wise.
        """
        self.A = torch.maximum(torch.zeros(Z.size()),Z)
        return self.A

    def backward(self, dA):
        """
        Perform the backward pass of the ReLU activation function.

        Args:
            dA (torch.Tensor): Gradient of the loss with respect to the output.

        Returns:
            dZ torch.Tensor: Gradient of the loss with respect to the input.
        """

        dZ = dA*torch.ones(dA.size())

        return dZ

    def update(self,lr):
        """
        ReLU does not have any parameters to update.
        """
        pass
