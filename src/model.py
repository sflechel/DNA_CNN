import torch
import torch.nn as nn


class DNACNN(nn.Module):
    def __init__(self, num_targets: int, seq_len: int):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=320, kernel_size=8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout1d(0.2),
            nn.Conv1d(in_channels=320, out_channels=480, kernel_size=8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout1d(0.2),
            nn.Conv1d(in_channels=480, out_channels=960, kernel_size=8),
            nn.ReLU(),
            nn.Dropout1d(0.5),
        )

        # we use a dummy input to get the output shape without doing the maths
        dummy_input = torch.zeros(1, 4, seq_len)
        flat_size = self.feature_extractor(dummy_input).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 925),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(925, num_targets),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # channels need to be the last dimension, if it's not we transpose
        if input.shape[-1] == 4:
            input = input.transpose(1, 2)

        features = self.feature_extractor(input)
        logits = self.classifier(features)

        return logits.squeeze(-1)

    def forward_return_hidden(
        self, input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass that return the output of the fully connected layer for regularization"""
        if input.shape[-1] == 4:
            input = input.transpose(1, 2)

        features = self.feature_extractor(input)
        flattened = self.classifier[0](features)

        hidden_weights = self.classifier[1](flattened)
        hidden_activation = self.classifier[2](hidden_weights)
        logits = self.classifier[3](hidden_activation)

        return logits.squeeze(-1), hidden_activation
