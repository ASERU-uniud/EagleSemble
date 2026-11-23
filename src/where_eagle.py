# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from json import load

import numpy as np
import shapely
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import optim

from data_parser import extract_road_features, adjust_road_array
from eagle_school import eagle_feeder, eagle_trainer, eagle_nester, eagle_tester#, AveragedPlateauEarlyStopping
from mini_transf_eagle import PositionalEncoding


class RoadSafetySequenceClassifier(nn.Module):
    def __init__(self, input_dim=3, d_model=64, nhead=4, num_layers=3, dropout=0.1, mean=0, std=1):
        super(RoadSafetySequenceClassifier, self).__init__()

        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)  # (Usa la classe definita in precedenza)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MODIFICA 1: Il classificatore ora agisce come un "valutatore di punto".
        # PyTorch applica automaticamente i layer Linear all'ultima dimensione (d_model),
        # mantenendo intatta la dimensione della sequenza.
        self.point_evaluator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # Output: 1 score di "pericolo" per punto
        )
        self.mean = mean
        self.std = std

    def forward(self, src_road):

        # src_road shape: (Batch, Seq_Len, 3)
        src_road = (src_road - self.mean) / self.std

        # Embedding + Posizione
        x = self.embedding(src_road)
        x = self.pos_encoder(x)

        # memory shape: (Batch, Seq_Len, d_model)
        memory = self.transformer_encoder(x)

        # MODIFICA 2: Nessun pooling. Passiamo l'intera sequenza al valutatore.
        # point_logits shape: (Batch, Seq_Len, 1)
        point_logits = self.point_evaluator(memory)

        # Rimuoviamo l'ultima dimensione per avere uno score puro per ogni punto
        # point_logits shape: (Batch, Seq_Len)
        point_logits = point_logits.squeeze(-1)

        # MODIFICA 3: Softmax lungo la sequenza (dim=1)
        # Ora per ogni batch, la somma delle probabilità lungo i punti della strada sarà esattamente 1.0
        # probabilities = torch.softmax(point_logits, dim=1)

        return point_logits


def get_oob_index(road_points, trajectory, oob: float, lane_width:int=5) -> int:
    center_line: shapely.LineString = shapely.LineString(coordinates=road_points)
    right_lane = center_line.buffer(distance=-lane_width, single_sided=True)
    ideal_trajectory: shapely.LineString | shapely.MultiLineString = center_line.parallel_offset(distance=lane_width/2)

    road_model = {
        'road_points': road_points,
        'center_line': center_line,
        'right_lane': right_lane,
        'ideal_trajectory': ideal_trajectory
    }

    for position in trajectory:
        if out_of_bounds(pos=position, rm=road_model, oob=oob):
            oob_index, _ = closest_point(pos=position, road_pts=road_points)
            return oob_index

    return -1


def out_of_bounds(pos: tuple, rm: dict, oob: float) -> bool:
    car_model = shapely.box(pos[0] - 1, pos[1] - 1, pos[0] + 1, pos[1] + 1)
    diff = car_model.difference(rm['right_lane'])
    actual_oob = diff.area / car_model.area
    return actual_oob > oob


def closest_point(pos: tuple, road_pts: list[tuple[float, float]]) -> tuple[int, float]:
    pos_np = np.array(pos)
    curr_min_dist = -1
    curr_min_dist_idx = -1
    for i, road_pt in enumerate(road_pts):
        road_pt_np = np.array(road_pt)
        norm = np.linalg.norm(road_pt_np - pos_np)
        if norm < curr_min_dist or curr_min_dist == -1:
            curr_min_dist = norm
            curr_min_dist_idx = i
    return curr_min_dist_idx, curr_min_dist

def where_eagle_parse(road, trajectory):
    road = adjust_road_array(np.array(road)) # len = 197
    label = [0] * (len(road) + 1) # len = 198
    oob_index = get_oob_index(road, trajectory, oob=0.5) % 198 # [-1, 196] % 198 = [0, 197]
    label[oob_index] = 1
    road = extract_road_features(road) # 196
    road = np.insert(road, 0, [0,0,0], axis=0) # len = 197
    road = np.insert(road, 0, [0,0,0], axis=0) # len = 198
    return road, oob_index

def where_eagle(roads, outcomes, trajectories, parse_fn:None=None, loss_fn=nn.CrossEntropyLoss,
                      mean=0, std=1,
                      train_size=0.89, verbose=True, model=None, nest=True, plot=True, version=''):

    if parse_fn is not None:
        roads, outcomes = eagle_feeder(parse_fn, roads, outcomes, trajectories, verbose=verbose)

    if model is None:
        model = RoadSafetySequenceClassifier(input_dim=roads.shape[-1], d_model=64, nhead=4, num_layers=3, mean=mean, std=std)

    loss_fn = loss_fn()
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.95)
    optimizer = optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5)
    scheduler = None

    model, history = eagle_trainer(roads, outcomes, model, optimizer, loss_fn=loss_fn, scheduler=scheduler,
                                   n_epochs=400, batch_size=500, decay=0, tolerance=1,
                                   # early_stopping=AveragedPlateauEarlyStopping(-0.001, 1, 20, 7),
                                   train_size=train_size, verbose=verbose)

    if nest:
        eagle_nester('../eagles/where_eagle', model, version, history)

    if plot:
        plt.plot(history)
        a = np.argsort(history)[0].item()
        plt.plot(a, history[a], 'r.')
        plt.show()

    return model

def y_parse(y_pred, y_test):
    y_pred = torch.argsort(y_pred, dim=1, descending=True)-1
    return y_pred, y_test

def only_failing_tcs(y_pred, y_test):
    failing_tcs, = (y_test!=197).nonzero(as_tuple=True)
    return y_pred[failing_tcs], y_test[failing_tcs]

def only_passing_tcs(y_pred, y_test):
    passing_tcs, = (y_test==197).nonzero(as_tuple=True)
    return y_pred[passing_tcs], y_test[passing_tcs]

def classify(y_pred, y_test):
    y_test = (y_test != 197).int()
    print(torch.argsort(y_pred, dim=1, descending=True)[:, 0])
    y_pred = (torch.argsort(y_pred, dim=1, descending=True)[:, 0] != 197).int()
    print(y_pred, sum(y_pred))
    return y_pred, y_test

def mean_abs_error(y_pred, y_test):
    y_pred = torch.argsort(y_pred, dim=1, descending=True)
    return torch.mean(torch.abs(y_pred[:, 0] - y_test).float()).item()

def mean_displacement(y_pred, y_test):
    y_pred = torch.argsort(y_pred, dim=1, descending=True)
    y_test = y_test.reshape(-1, 1)
    return torch.mean(torch.argmin(torch.abs(y_pred - y_test), dim=1).float()).item()

def mean_displacement_with_neighbors(y_pred, y_test):
    y_pred1 = y_pred.clone()
    # y_pred1[:, 2:-1] += y_pred1[:, 1:-2] + y_pred1[:, 3:] # -1 goes to 0 (add 1)
    # y_pred1[:, 1:-2] += y_pred[:, 0:-3] + y_pred[:, 2:-1] # -1 goes to 197 (mod 198)
    y_pred1[:, 0:-2] += y_pred[:, 1:-1]; y_pred1[:, 1:-1] += y_pred[:, 0:-2]
    y_pred1 = torch.argsort(y_pred1, dim=1, descending=True)
    y_test = y_test.reshape(-1, 1)
    return torch.mean(torch.argmin(torch.abs(y_pred1 - y_test), dim=1).float()).item()


class MinimizeEagle(nn.Module):
    def __init__(self, minimizer_model, classifier_model):
        super(MinimizeEagle, self).__init__()
        self.minimizer_model = minimizer_model
        self.classifier_model = classifier_model

    def forward(self, x):
        logits = torch.softmax(self.minimizer_model(x), dim=1).clone()
        # outcomes = torch.sigmoid(self.classifier_model(x[:, 2:, :])).clone()
        # succeeding_tcs, = (outcomes.squeeze() > .53).nonzero(as_tuple=True)
        # logits[succeeding_tcs, 197] = 1
        return logits

if __name__ == '__main__':
    with open('../data/sensodat.json') as f:
        roads, _, _, trajectories = load(f)

    data, labels = eagle_feeder(where_eagle_parse, roads, trajectories)
    labels = labels.long()

    torch.save(data, f'../eagles/where_eagle/data/where_eagle_data.pt')
    torch.save(labels, f'../eagles/where_eagle/data/where_eagle_labels_mod.pt')

    # data = torch.load('../eagles/where_eagle/data/where_eagle_data.pt')
    # labels = torch.load('../eagles/where_eagle/data/where_eagle_labels_mod.pt')

    # data, labels = only_passing_tcs(data, labels) # Test Case SELECTION

    # data = (data-torch.mean(data, dim=(0,1))/torch.std(data, dim=(0,1))) # STANDARD SCALER FOR ROADS

    (data_train, data_test,
     labels_train, labels_test, ) = train_test_split(data, labels, train_size=0.90, shuffle=True, random_state=42)

    # from mini_transf_eagle import RoadSafetyClassifier
    # classifier = torch.load('../eagles/mini_transf_eagle/model-d64h4b100.pt', weights_only=False)

    model = where_eagle(data_train, labels_train, None, version='5', nest=False)
    # minimizer = torch.load('../eagles/where_eagle/model5.pt', weights_only=False)

    # model = MinimizeEagle(minimizer, classifier)
    # model = minimizer

    metrics = {
        'CrossEntropyLoss': (lambda loss_fn: lambda y_true, y_pred: loss_fn()(y_true, y_pred).detach().item())(nn.CrossEntropyLoss),
        'Mean Displacement': mean_displacement,
        'Mean Displacement with neighbors': mean_displacement_with_neighbors,
        'Mean Absolute Error': mean_abs_error,
    }

    classification_metrics = {
        'Precision': lambda y_pred, y_test: float(sum(torch.logical_and(y_pred, y_test))) / float( sum(y_pred)),
        'Accuracy': lambda y_pred, y_test: float( len(y_pred) - sum(torch.logical_xor(y_pred, y_test))) / float(len(y_pred)),
        'Recall': lambda y_pred, y_test: float(sum(torch.logical_and(y_pred, y_test))) / float(sum(y_test)),
        'Selected TCs': lambda y_pred, y_test: int(sum(y_pred)),
    }


    stats, _ = eagle_tester(model, None, data_test,
                            labels_test, y_parse_fn=classify,
                            batch_size=128, **metrics)

    for k, v in stats.items():
        print(k + ': ', v)