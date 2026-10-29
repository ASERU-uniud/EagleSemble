from json import load, dump

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import optim

from eagle_school import eagle_feeder, eagle_tester, eagle_trainer
from mini_transf_eagle import mini_transf_eagle_parse, mini_transf_eagle, RoadSafetyClassifier, PositionalEncoding, \
    eagle_raw_parse
from definitions import seq_eagle_model
from data_parser import adjust_road_array, extract_road_features

from roadfury_parsing import compute_features
from src.roadfury import road_fury


def mini_eagle_parse(road, outcome):
    road = adjust_road_array(np.array(road, dtype=np.float32))
    road = extract_road_features(road)
    return road.reshape(-1), [1 - outcome[0]]

def mini_eagle(roads, outcomes, parse_fn:None=mini_eagle_parse, loss_fn=torch.nn.BCELoss, verbose=True):

    if parse_fn is not None:
        roads, outcomes = eagle_feeder(parse_fn, roads, outcomes, verbose=verbose)
        roads, outcomes = roads.to('cuda'), outcomes.to('cuda')

    model = seq_eagle_model(roads.shape[1], outcomes.shape[1], 6).to('cuda')

    loss_fn = loss_fn().to('cuda')
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    model, history = eagle_trainer(roads, outcomes, model, optimizer, loss_fn=loss_fn, scheduler=None,
                                   n_epochs=15000, batch_size=None, decay=0.5, tolerance=1000,
                                   )
    return model


class Ensemble(torch.nn.Module):
    def __init__(self, voting_criteria, *models):
        super(Ensemble, self).__init__()
        self.models = models
        self.voting_criteria = voting_criteria

    def forward(self, x):
        results = [model(x) for model in self.models]
        return self.voting_criteria(results)

def time_to_first_fault(outcomes, durations):
    failing_tcs, = outcomes.nonzero(as_tuple=True)
    tff = sum(durations[:failing_tcs[0]+1])
    return tff.item()

def time_to_last_fault(outcomes, durations):
    failing_tcs, = outcomes.nonzero(as_tuple=True)
    tlf = sum(durations[:failing_tcs[-1]+1])
    return tlf.item()

def apfd(outcomes, _):
    failing_tcs, = outcomes.nonzero(as_tuple=True)
    apfd = 1 - sum(failing_tcs+1)/(len(failing_tcs)*len(outcomes)) + 1/(2*len(outcomes))
    return apfd.item()

def apfdc(outcomes, durations):
    failing_tcs, = outcomes.nonzero(as_tuple=True)
    apfdc = 1 - sum([sum(durations[:i+1]) for i in failing_tcs])/(len(failing_tcs)*sum(durations)) + 1/(2*len(failing_tcs))
    return apfdc.item()

def duration_parse(road, outcome, norm_duration):
    road = extract_road_features(adjust_road_array(np.array(road)))
    label = -norm_duration/2+1 if outcome[0] == 1 else -norm_duration/2+0.5
    return road, [label]

def road_fury_parse(road, outcome):
    road = compute_features(road)
    return road, outcome

if __name__ == '__main__':

    with open('../data/sensodat.json') as f:
        roads, outcomes, durations, _ = load(f)

    outcomes = 1 - torch.tensor(outcomes, dtype=torch.float32).reshape(-1,1)
    durations = torch.tensor(durations, dtype=torch.float32)
    durations = durations.squeeze()
    # norm_durations = torch.nn.functional.sigmoid((durations.argsort(dim=0).argsort(dim=0)/(durations.shape[0]-1)))
    roads, labels = eagle_feeder( road_fury_parse,roads, outcomes) #,  norm_durations)

    roads = roads.transpose(1, 2)
    labels = labels.squeeze()

    # print(min(durations[outcomes.nonzero(as_tuple=True)[0]]), sum(outcomes)/len(outcomes))
    # print(sorted(durations[outcomes.nonzero(as_tuple=True)[0]], key=lambda x: x, reverse=False)[:20])

    # roads = roads.reshape(roads.shape[0], -1, 3)
    roads_mean, roads_std = torch.mean(roads, dim=(0, 1)), torch.std(roads, dim=(0, 1))
    # roads = (roads - roads_mean) / roads_std
    # roads = roads.reshape(roads.shape[0], -1)

    for index_train, index_test in KFold(n_splits=10).split(roads):

        (roads_train, roads_test,
         labels_train, outcomes_test,
         durations_test)       =        (roads[index_train], roads[index_test],
                                         labels[index_train], outcomes[index_test],
                                         durations[index_test])


        model1=road_fury(roads_train, labels_train, parse_fn=None, nest=False, plot=False)
        model2=road_fury(roads_train, labels_train, parse_fn=None, nest=False, plot=False)
        model3=road_fury(roads_train, labels_train, parse_fn=None, nest=False, plot=False)
        model1, model2, model3 = (torch.nn.Sequential(model1, torch.nn.Sigmoid()),
                                  torch.nn.Sequential(model2, torch.nn.Sigmoid()),
                                  torch.nn.Sequential(model3, torch.nn.Sigmoid()))
        model = Ensemble(lambda x: sum(x)/len(x), model1, model2, model3)

        # model = mini_eagle(roads_train, outcomes_train, parse_fn=None)

        # model = torch.load('../eagles/mini_transf_eagle/model-d16h4b100.pt', weights_only=False)
        # model = torch.nn.Sequential(model, torch.nn.Sigmoid().to('cuda')).to('cuda')

        def y_parse(y_pred, y_test):
            result = torch.column_stack((y_pred.squeeze(), y_test))
            result = torch.stack(sorted(result, key=lambda x: x[0], reverse=True))
            return result[:, 1], result[:, 2]

        metrics = {
            'time to first fault': time_to_first_fault,
            'time to last fault': time_to_last_fault,
            'apfd': apfd,
            'apfdc': apfdc,
        }

        stats, _ = eagle_tester(model, None, roads_test,
                                torch.column_stack((outcomes_test.squeeze(), durations_test)), y_parse_fn=y_parse, batch_size=128, **metrics)

        for k, v in stats.items():
            print(k+': ', v)
        with open('stats/roadfury_stats.csv', 'a') as f:
            f.write(','.join([str(s) for s in stats.values()]) + '\n')
