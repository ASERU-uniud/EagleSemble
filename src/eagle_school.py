import signal
from copy import deepcopy
from json import dump
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from numpy import inf

__stop_training = False
def signal_handler(_, __):
    global __stop_training
    __stop_training = True
signal.signal(signal.SIGINT, signal_handler)

def eagle_trainer(data, labels, model, optimizer,
                  scheduler=None,
                  swa = 0,
                  train_size=0.7,
                  random_state=42,
                  loss_fn=nn.MSELoss(),
                  n_epochs=10000,
                  batch_size=None,
                  eval_batch_size=None,
                  decay=0.0,
                  tolerance=0.1,
                  verbose=True,
                  ):

    global __stop_training
    __stop_training = False

    # train-validation split for validation during training
    x_train, x_validation, y_train, y_validation = train_test_split(data, labels,
                                                                    train_size=train_size,
                                                                    shuffle=True,
                                                                    random_state=random_state)

    model, loss_fn = model.to('cuda'), loss_fn.to('cuda')

    if 0 < swa < 1:
        swa_model = AveragedModel(model)
        # swa_scheduler = SWALR(optimizer, swa_lr=0.005)
        if swa < 1:
            swa_epoch_start =  (1-swa)*n_epochs
        else:
            swa_epoch_start = swa
    else:
        swa_model = model
        swa_epoch_start = n_epochs


    if batch_size is None:
        batch_size = len(x_train)  # size of each batch
    elif batch_size < 1:
        batch_size = int(round(batch_size*len(x_train)))
    batch_start = range(0, len(x_train), batch_size)

    if eval_batch_size is None:
        eval_batch_size = batch_size
    elif eval_batch_size < 1:
        eval_batch_size = int(round(eval_batch_size*len(x_validation)))
    eval_batch_start = range(0, len(x_validation), eval_batch_size)

    # Hold the best model
    best_loss = inf
    best_weights = deepcopy(model.state_dict())
    history = []

    loss_name = loss_fn.__class__.__name__[:-4]
    n_epochs_digits = len(str(n_epochs))

    if tolerance <= 1:
        tolerance = tolerance*n_epochs
    fail_counter = 0.0

    for epoch in range(n_epochs):
        model.train()
        bar = tqdm(total=len(batch_start), unit="batch", mininterval=0, disable=(not verbose), leave=True, dynamic_ncols=True)
        bar.set_description(f"Epoch {epoch:{n_epochs_digits}}")
        for start in batch_start:
            if __stop_training:
                break

            # take a batch
            x_batch = x_train[start:start+batch_size].to('cuda')
            y_batch = y_train[start:start+batch_size].to('cuda')
            # forward pass
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress
            bar.set_postfix_str(f'Training {loss_name}: {loss.item():.4g}')
            bar.update()
        # evaluate accuracy at end of each epoch
        bar.refresh()

        model.eval()

        if __stop_training:
            break

        bar.reset()
        n_eval_batch = len(eval_batch_start)
        bar.total = n_eval_batch
        loss = 0
        with torch.inference_mode():
            for start in eval_batch_start:
                y_pred = model(x_validation[start:start+eval_batch_size].to('cuda'))
                batch_loss = loss_fn(y_pred, y_validation[start:start+eval_batch_size].to('cuda')).item()
                loss += batch_loss
                bar.set_postfix_str(f'Validation {loss_name}: {batch_loss:.4g}')
                bar.update()
        loss = loss / n_eval_batch

        history.append(loss)
        bar.set_postfix_str(f'Validation {loss_name}: {loss:.4g}')


        if epoch > swa_epoch_start:
            swa_model.update_parameters(model)
            # swa_scheduler.step()
        else:
            if scheduler is not None:
                scheduler.step(metrics=loss)

        if loss < best_loss:
            fail_counter *= decay
            best_loss = loss
            best_weights = deepcopy(model.state_dict())

            bar.set_postfix_str(bar.postfix + '  *****')
        else:
            if fail_counter > tolerance:
                break
            else:
                fail_counter += 1.0


        bar.update(0)
        bar.close()

    if swa == 0:
        model.load_state_dict(best_weights)

    return swa_model, history


def eagle_feeder(parse_fn, *data, verbose=True, message='Parsing data', output_type=torch.Tensor):
    assert all([len(column) == len(data[0]) for column in data])
    n_rows = len(data[0])
    x, y = [[]]*n_rows, [[]]*n_rows
    with tqdm(range(n_rows), unit="TestCase", mininterval=0, disable=(not verbose)) as bar:
        bar.set_description(message)
        for i in bar:
            x[i], y[i] = parse_fn(*[d[i] for d in data])

    x, y = np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)
    if output_type == torch.Tensor:
        x, y = torch.from_numpy(x), torch.from_numpy(y)
    elif output_type is not None:
        x, y = output_type(x), output_type(y)

    return x, y


def eagle_tester(model, parse_fn, *data, y_parse_fn=None, batch_size=None, verbose=True, **metrics):

    if parse_fn is not None:
        x_test, y_test = eagle_feeder(parse_fn, *data, verbose=verbose)
    else:
        x_test, y_test = data
    y_test = y_test.to('cuda')

    model.eval()

    if batch_size is None:
        batch_size = len(x_test)
    elif batch_size < 1:
        batch_size *= int(round(batch_size*len(x_test)))

    batch_start = range(0, len(x_test), batch_size)

    y_pred = torch.Tensor([]).to('cuda')
    with torch.inference_mode():
        for start in batch_start:
            batch = x_test[start:start+batch_size].to('cuda')
            prediction = model(batch)
            y_pred = torch.cat((y_pred, prediction))

    y_pred, y_test = y_pred.detach(), y_test.detach()

    if y_parse_fn is not None:
        y_pred, y_test = y_parse_fn(y_pred, y_test)

    return {name: metric(y_pred, y_test) for name, metric in metrics.items()}, y_pred


def eagle_nester(path, model, model_number='', history=None, stats=None, test_set=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    torch.save(model, str(path)+'/model'+model_number+'.pt')
    if history is not None:
        with open(str(path)+'/model-history'+model_number+'.json', 'w') as f:
            dump(history, f)
    if test_set is not None:
        with open(str(path)+'/test_set'+model_number+'.json', 'w') as f:
            dump(test_set, f)
    if stats is not None:
        with open(str(path)+'/stats.csv', 'a') as fp:
            fp.write(','.join([str(s) for s in stats]) + '\n')

def export_to_onnx(model, input_dims, filename):
    model = model.to('cpu')
    model.eval()
    onnx_program = torch.onnx.export(model, (torch.randn(*input_dims),))
    onnx_program.save(filename)