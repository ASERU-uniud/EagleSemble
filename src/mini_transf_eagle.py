from json import load

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import optim
import math

from data_parser import adjust_road_array, extract_road_features
from eagle_school import eagle_trainer, eagle_feeder, eagle_tester, eagle_nester


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()

        # Crea matrice di zeri (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Calcola le frequenze per seno e coseno
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Applica seno agli indici pari, coseno ai dispari
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Aggiunge la dimensione del batch: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # register_buffer salva 'pe' nello state_dict ma non lo rende addestrabile (non aggiorna i pesi)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, d_model)
        # Somma il Positional Encoding ai dati
        x = x + self.pe[:, :x.size(1), :]
        return x

class RoadSafetyClassifier(nn.Module):
    def __init__(self, input_dim=3, d_model=64, nhead=4, num_layers=3, dropout=0.1, mean=0, std=1):
        super(RoadSafetyClassifier, self).__init__()

        # 1. Proiezione Input (es. 3 feature -> 64)
        self.embedding = nn.Linear(input_dim, d_model)

        # 2. Positional Encoding (Usa la stessa classe di prima!)
        self.pos_encoder = PositionalEncoding(d_model)

        # 3. Solo l'Encoder del Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True  # Fondamentale
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Testa di Classificazione (MLP finale)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # Output singolo (Sicura/Non Sicura)
        )
        self.mean = mean
        self.std = std

    def forward(self, src_road):
        # src_road shape: (Batch, Seq_Len, 3)
        src_road = (src_road - self.mean) / self.std
        # Embedding + Posizione
        x = self.embedding(src_road)
        x = self.pos_encoder(x)

        # Passaggio nell'Encoder
        # memory shape: (Batch, Seq_Len, d_model)
        memory = self.transformer_encoder(x)

        # AGGREGAZIONE (Global Average Pooling)
        # Facciamo la media lungo la dimensione della sequenza (dim=1)
        # x_pooled shape: (Batch, d_model)
        x_pooled = memory.mean(dim=1)

        # Classificazione finale
        # logits shape: (Batch, 1)
        logits = self.classifier(x_pooled)

        return logits

def mini_transf_eagle_parse(road, outcome):
    road = extract_road_features(adjust_road_array(np.array(road)))
    return road, [1-outcome[0]]

def eagle_raw_parse(road, outcome):
    road = adjust_road_array(np.array(road))
    return road, [1 - outcome[0]]

def mini_transf_eagle(roads, outcomes, parse_fn:None=mini_transf_eagle_parse, loss_fn=nn.BCEWithLogitsLoss,
                      mean=0, std=1,
                      train_size=0.89, verbose=True, model=None, nest=True, plot=True, version=''):

    if parse_fn is not None:
        roads, outcomes = eagle_feeder(parse_fn, roads, outcomes, verbose=verbose)

    if model is None:
        model = RoadSafetyClassifier(input_dim=roads.shape[-1], d_model=128, nhead=8, num_layers=4, mean=mean, std=std)

    loss_fn = loss_fn()
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.9)
    optimizer = optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5)

    model, history = eagle_trainer(roads, outcomes, model, optimizer, loss_fn=loss_fn, scheduler=None,
                                   n_epochs=400, batch_size=200, decay=0.5, tolerance=35,
                                   train_size=train_size, verbose=verbose)

    if nest:
        eagle_nester('../eagles/mini_transf_eagle', model, version, history)

    if plot:
        plt.plot(history)
        plt.show()
    return model


# class MCDropout(torch.nn.Dropout):
#     def forward(self, x):
#         return torch.nn.functional.dropout(
#             x,
#             self.p,
#             training=True
#         )
# class NoiseLayer(torch.nn.Module):
#     def __init__(self, std=0.1):
#         super().__init__()
#         self.std = std
#
#     def forward(self, x):
#         noise = torch.randn_like(x) * self.std
#         return x + noise


if __name__ == '__main__':

    with open('../data/sensodat.json') as f:
        roads, outcomes, _, _ = load(f)

    roads, outcomes = eagle_feeder( mini_transf_eagle_parse,roads, outcomes)

    roads_mean, roads_std = torch.mean(roads, dim=(0, 1)), torch.std(roads, dim=(0, 1))
    # roads = (roads - roads_mean) / roads_std

    print('Splitting train and test...', end='')
    (roads_train, roads_test,
     outcomes_train, outcomes_test,) = train_test_split(roads, outcomes, train_size=0.90, shuffle=True)
    print('Done.')

    model = None

    # model = torch.load('../eagles/mini_transf_eagle/model-d16h4b100.pt', weights_only=False)

    model=mini_transf_eagle(roads_train, outcomes_train, model=model, version='-d16h4b100', parse_fn=None, mean=roads_mean.cuda(), std=roads_std.cuda())

    loss_fn = nn.BCELoss().to('cuda')
    sigm = nn.Sigmoid().to('cuda')

    loss_metric = lambda loss_fn: lambda y_pred, y_test: float(loss_fn(sigm(y_pred), y_test).detach())
    y_parse_fn = lambda y: torch.round(sigm(y).detach()).int()
    metrics = {
        loss_fn.__class__.__name__[:-4] + ' on test set': loss_metric(loss_fn),
        'Precision': lambda y_pred, y_test: float(sum(torch.logical_and(y_parse_fn(y_pred), y_test))) / float(
            sum(y_parse_fn(y_pred))),
        'Accuracy': lambda y_pred, y_test: float(
            len(y_pred) - sum(torch.logical_xor(y_parse_fn(y_pred), y_test))) / float(len(y_pred)),
        'Recall': lambda y_pred, y_test: float(sum(torch.logical_and(y_parse_fn(y_pred), y_test))) / float(sum(y_test)),
        'Selected TCs': lambda y_pred, y_test: int(sum(y_parse_fn(y_pred))),
    }

    stats, _ = eagle_tester(model, None, roads_test, outcomes_test, batch_size=128, **metrics)

    for k, v in stats.items():
        print(k+': ', v)

    # print(roads_mean.tolist(), roads_std.tolist())

    # with open('test_set.json', 'w') as f:
    #     dump(icst_syntax(roads_test, outcomes_test), f)

    # export_model = nn.Sequential(model.to('cpu'), nn.Sigmoid())
    # export_model.eval()
    #
    # onnx_program = torch.onnx.export(export_model, (torch.randn(1, 196, 3),))
    #
    # onnx_program.save('eagle_semble/mini_eagle_norm.onnx')

