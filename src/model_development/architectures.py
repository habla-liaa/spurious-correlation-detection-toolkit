import torch.nn as nn


class Wav2Vec2Classifier(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.layer_pooling = nn.Linear(kwargs['num_layers'], 1)
        if not kwargs.get('without_hidden', False):
            self.projection = nn.Sequential(
                nn.Linear(kwargs['in_channels'], kwargs['projection_dim']),
                nn.ReLU(), 
                nn.Dropout(kwargs.get('dropout', 0.1)),
            )
            projection_dim = kwargs['projection_dim']
        else:
            self.projection = None
            projection_dim = kwargs['in_channels']
        
        self.classifier = nn.Linear(projection_dim, kwargs['num_classes'])

    def forward(self, hidden_states):
        pooled_states = self.layer_pooling(hidden_states).squeeze(-1)  
        if self.projection is not None:
            projected = self.projection(pooled_states)
        else:
            projected = pooled_states
        logits = self.classifier(projected)
        return logits


class AcousticCNNClassifier(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        multiplier = kwargs.get('channel_multiplier', 2)
        cnn_channels = int(kwargs['in_channels'] * multiplier)
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=kwargs['in_channels'], 
                      out_channels=cnn_channels, 
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True)
        )

        self.projection = nn.Sequential(
            nn.Linear(cnn_channels, kwargs['projection_dim']),
            nn.ReLU(), 
            nn.Dropout(kwargs.get('dropout', 0.1)),
        )
        
        self.classifier = nn.Linear(kwargs['projection_dim'], kwargs['num_classes'])

    def forward(self, acoustic_features):
        x = self.conv_block(acoustic_features)
        pooled_features = x.mean(dim=-1)        
        projected = self.projection(pooled_features)
        logits = self.classifier(projected)
        return logits
