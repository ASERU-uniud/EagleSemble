from json import load
from torch import optim
from eagle_school import *
from data_parser import *
# from sensodat_loader import icst_syntax
from matplotlib import pyplot as plt
from definitions import *
from definitions import select
from eagle_school import eagle_feeder, eagle_tester, eagle_nester


def behav_eagle_parse(road, traj):
    road = adjust_road_array(np.array(road, dtype=np.float32))
    traj = adjust_traj_array(np.array(traj, dtype=np.float32))
    traj = extract_traj_features(traj, road)
    road = extract_road_features(road)
    return road.reshape(-1), traj.reshape(-1)

def behav_eagle(roads, trajectories, parse_fn:None=behav_eagle_parse, loss_fn=torch.nn.HuberLoss, verbose=True):

    if parse_fn is not None:
        roads, trajectories = eagle_feeder(parse_fn, roads, trajectories, verbose=verbose)
        roads, trajectories = roads.to('cuda'), trajectories.to('cuda')

    model = behav_eagle_model(roads.shape[1], trajectories.shape[1]).to('cuda')

    loss_fn = loss_fn().to('cuda')
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.9)

    model, history = eagle_trainer(roads, trajectories, model, optimizer, loss_fn=loss_fn, scheduler=scheduler,
                                   n_epochs=15000, batch_size=0.1, decay=0.5, tolerance=600,
                                   )
    return model, history, loss_fn

def mini_geom_eagle_parse(road, trajectory, outcome):
    road = adjust_road_array(np.array(road, dtype=np.float32))
    traj = adjust_traj_array(np.array(trajectory, dtype=np.float32), target_size=197)
    traj = extract_traj_features(traj, road)
    road = extract_road_features(road)
    return np.concatenate((road.reshape(-1), traj.reshape(-1))), [1-outcome[0]]

def mini_geom_eagle(roads, trajectories, outcomes, parse_fn:None=mini_geom_eagle_parse, loss_fn=torch.nn.BCELoss, verbose=True):

    if parse_fn is not None:
        roads, outcomes = eagle_feeder(parse_fn, roads, trajectories, outcomes, verbose=verbose)
        roads, outcomes = roads.to('cuda'), outcomes.to('cuda')

    model = seq_eagle_model(roads.shape[1], outcomes.shape[1], 6).to('cuda')

    loss_fn = loss_fn().to('cuda')
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    model, history = eagle_trainer(roads, outcomes, model, optimizer, loss_fn=loss_fn, scheduler=None,
                                   n_epochs=15000, batch_size=None, decay=0.5, tolerance=1000,
                                   )
    return model, history, loss_fn


def geom_eagle_parse(road, outcome):
    road = adjust_road_array(np.array(road, dtype=np.float32))
    road = extract_road_features(road)
    return road.reshape(-1), [1 - outcome[0]]

if __name__ == '__main__':

    print('Loading dataset... ', end='')
    with open('../data/sensodat.json') as f:
        roads, outcomes, durations, trajectories = load(f)
    print('Done.')

    print('Selecting relevant TCs...', end='')
    roads, trajectories, outcomes = select(roads, trajectories, outcomes, where=lambda row: len(row[1]) > 11)
    print('Done.')

    print('Splitting data...', end='')
    (roads_train, roads_test,
     traj_train, traj_test,
     out_train, out_test) = train_test_split(roads, trajectories, outcomes, train_size=0.9, shuffle=True)
    print('Done.')

    # print('Saving test set...', end='')
    # test_set = icst_syntax(X_test, y_test)
    # print('Done.')

    loss_metric = lambda loss_fn: lambda y_pred, y_test: float(loss_fn(y_pred, y_test).detach())

    # trained_behav_eagle, history, loss_fn = behav_eagle(roads_train, traj_train)
    # eagle_nester('behav_eagle', trained_behav_eagle, history=history)
    # loss_on_test = eagle_tester(trained_behav_eagle, behav_eagle_parse, roads_test, traj_test, Huber=loss_metric(loss_fn))
    # print(f'HuberLoss on test set: {loss_on_test['Huber']:.4f}')
    #
    # X_train, y_train = eagle_feeder(geom_eagle_parse, roads_train, out_train)
    # X_train = torch.column_stack((X_train, trained_behav_eagle(X_train).detach()))
    # trained_m_geom_eagle, _, loss_fn = mini_geom_eagle(X_train, None, y_train, parse_fn=None)
    # best_loss = min(history)

    trained_m_geom_eagle, history, loss_fn = mini_geom_eagle(roads_train, traj_train, out_train)

    y_parse_fn = lambda y: torch.round(y.detach()).int()
    metrics = {
        loss_fn.__class__.__name__[:-4] + ' on test set': loss_metric(loss_fn),
        'Precision' : lambda y_pred, y_test: float(sum(torch.logical_and(y_parse_fn(y_pred), y_test))) / float(sum(y_parse_fn(y_pred))),
        'Accuracy' : lambda y_pred, y_test: float(len(y_pred) - sum(torch.logical_xor(y_parse_fn(y_pred), y_test))) / float(len(y_pred)),
        'Recall' : lambda y_pred, y_test: float(sum(torch.logical_and(y_parse_fn(y_pred), y_test))) / float(sum(y_test)),
        'Selected TCs' : lambda y_pred, y_test: int(sum(y_parse_fn(y_pred))),
    }
    # print("\nBCE on validation set: %.4f" % best_loss)

    # X_test, y_test = eagle_feeder(geom_eagle_parse, roads_test, out_test)
    # X_geom = torch.column_stack((X_test, trained_behav_eagle(X_test)))
    stats, _ = eagle_tester(trained_m_geom_eagle, mini_geom_eagle_parse, roads_test, traj_test, out_test, **metrics)
    for k, v in stats.items():
        print(k+': ', v)

    eagle_nester('../eagles/transf_eagle', trained_m_geom_eagle, 'classifier')

    plt.plot(history)
    plt.show()