import numpy as np
import torch.nn as nn
import torch
from matplotlib import pyplot as plt
from torch import optim

from src.eagle_school import eagle_feeder, eagle_trainer, eagle_nester
from src.roadfury_parsing import compute_features


class RoadFury(nn.Module):
    """
    Transformer Encoder for road geometry classification.

    Input: [B, 10, 197] → permute → [B, 197, 10]
    → Linear projection to d_model
    → Prepend learnable [CLS] token + positional embedding
    → Pre-LN TransformerEncoder (4 layers, 8 heads)
    → CLS-only pooling → MLP classifier → logit
    """
    def __init__(self, in_channels=10, seq_len=197, d_model=128,
                 nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: [B, C, L] → [B, L, C]
        x = x.permute(0, 2, 1)
        B, L, C = x.shape
        x = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding[:, :L + 1, :]
        x = self.transformer(x)
        return self.classifier(x[:, 0, :]).squeeze(-1)

def road_fury_parse(road, outcome):
    road = compute_features(road)
    return road, outcome

def road_fury(roads, outcomes, parse_fn:None=road_fury_parse, loss_fn=nn.BCEWithLogitsLoss,
                      train_size=0.89, verbose=True, model=None, nest=True, plot=True, version=''):

    if parse_fn is not None:
        roads, outcomes = eagle_feeder(parse_fn, roads, outcomes, verbose=verbose)

    if model is None:
        model = RoadFury()

    loss_fn = loss_fn()
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.9)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)

    model, history = eagle_trainer(roads, outcomes, model, optimizer, loss_fn=loss_fn, scheduler=None,
                                   n_epochs=75, batch_size=100,
                                   swa=49,
                                   decay=0, tolerance=np.inf,
                                   train_size=train_size, verbose=verbose)

    if nest:
        eagle_nester('../eagles/mini_transf_eagle', model, version, history)

    if plot:
        plt.plot(history)
        plt.show()
    return model
