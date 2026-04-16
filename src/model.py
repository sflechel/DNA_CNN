import torch
import torch.nn as nn


class DNACNN(nn.Module):
    def __init__(self, seq_len: int = 1000):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=15),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
        )

        # we use a dummy input to get the output shape without doing the maths
        dummy_input = torch.zeros(1, 4, seq_len)
        flat_size = self.feature_extractor(dummy_input).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Dropout(
                0.5
            ),  # randomly drop half the feature neurons to prevent overfitting on any specific feature
            nn.Linear(128, 1),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # channels need to be the last dimension, if it's not we transpose
        if input.shape[-1] == 4:
            input = input.transpose(1, 2)

        features = self.feature_extractor(input)
        logits = self.classifier(features)

        return logits.squeeze(-1)
