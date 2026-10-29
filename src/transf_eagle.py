import signal
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.model_selection import train_test_split

from data_parser import adjust_road_array, adjust_traj_array, extract_road_features, frenet_frame
from src.eagle_school import eagle_feeder, eagle_nester


# ==========================================
# 1. POSITIONAL ENCODING
# ==========================================
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


def get_causal_mask(sz):
    # Genera la maschera triangolare per il decoder (Look-ahead mask)
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# ==========================================
# 2. IL MODELLO TRANSFORMER
# ==========================================
class TrajectoryTransformer(nn.Module):
    def __init__(self, input_dim=2, output_dim=2, d_model=32, nhead=4, num_layers=3, dropout=0.1):
        super(TrajectoryTransformer, self).__init__()

        # Proiezioni lineari per portare (x,y) alla dimensione del Transformer
        self.encoder_embedding = nn.Linear(input_dim, d_model)
        self.decoder_embedding = nn.Linear(output_dim, d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Core Transformer (batch_first=True è fondamentale per avere shape (N, S, E))
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )

        # Proiezione finale per tornare a (x,y)
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, src_road, tgt_traj, tgt_mask=None):
        # 1. Embedding e Positional Encoding per la Strada (Encoder)
        src = self.encoder_embedding(src_road)
        src = self.pos_encoder(src)

        # 2. Embedding e Positional Encoding per la Traiettoria (Decoder)
        tgt = self.decoder_embedding(tgt_traj)
        tgt = self.pos_encoder(tgt)

        # 3. Passaggio nel Transformer
        out = self.transformer(src, tgt, tgt_mask=tgt_mask)

        # 4. Proiezione finale alle coordinate
        out = self.output_layer(out)
        return out


# ==========================================
# 3. METRICHE E LOSS
# ==========================================
def compute_metrics(pred, target):
    """Calcola Average Displacement Error e Final Displacement Error"""
    # Pred e Target shape: (Batch, Seq_Len, 2)
    error = torch.sqrt(torch.sum((pred - target) ** 2, dim=-1))  # Distanza Euclidea (Batch, Seq_Len)
    ade = torch.mean(error)
    fde = torch.mean(error[:, -1])
    return ade.item(), fde.item()


# ==========================================
# 4. CICLO DI TRAINING (Un'epoca)
# ==========================================
def train_epoch(model, optimizer, criterion, road_data, traj_data, device):
    model.train()
    optimizer.zero_grad()

    road_data = road_data.to(device)
    traj_data = traj_data.to(device)

    # TEACHER FORCING:
    # Input al decoder: tutta la traiettoria tranne l'ultimo punto
    tgt_input = traj_data[:, :-1, :]
    # Target atteso: tutta la traiettoria tranne il primo punto
    tgt_expected = traj_data[:, 1:, :]

    # Genera maschera causale per la sequenza di input del decoder
    tgt_mask = get_causal_mask(tgt_input.size(1)).to(device)

    # Forward pass
    predictions = model(road_data, tgt_input, tgt_mask)

    # Calcolo Loss e Metriche
    loss = criterion(predictions, tgt_expected)
    ade, fde = compute_metrics(predictions, tgt_expected)

    # Backward pass e ottimizzazione
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Previene gradienti esplosivi
    optimizer.step()

    return loss.item(), ade, fde


# ==========================================
# 5. INFERENZA AUTOREGRESSIVA (Valutazione/Test)
# ==========================================
def predict_trajectory(model, road_data, start_point, max_len, device):
    """Genera la traiettoria punto dopo punto partendo dalla strada e dal punto iniziale"""
    model.eval()
    road_data = road_data.to(device)

    # Il decoder inizia con la posizione attuale dell'auto (es. 0,0)
    # Shape: (Batch_size, 1, 2)
    current_tgt = start_point.unsqueeze(1).to(device)

    with torch.no_grad():
        for _ in range(max_len):
            # Crea la maschera per la sequenza generata fino ad ora
            tgt_mask = get_causal_mask(current_tgt.size(1)).to(device)

            # Forward pass
            predictions = model(road_data, current_tgt, tgt_mask)

            # Prendi l'ultimo punto predetto (Batch_size, 1, 2)
            next_point = predictions[:, -1:, :]

            # Concatena il nuovo punto alla sequenza in ingresso al decoder
            current_tgt = torch.cat([current_tgt, next_point], dim=1)

    # Ritorna la traiettoria generata (escludendo il punto di start iniziale se desiderato)
    return current_tgt

from json import load
from copy import deepcopy
from definitions import *

def human_time(seconds:int):
    s = seconds % 60
    m = seconds // 60
    h = m // 60
    d = h // 24
    output =  ((f'{d}d ' if d > 0 else '') +
               (f'{h}h ' if h>0 or d>0 else '') +
               (f'{m}m ' if h>0 or d>0 or m>0 else '') +
                f'{s}s')
    return output
# ==========================================
# 6. BLOCCO DI TEST (Main)
# ==========================================
if __name__ == "__main__":

    # Iperparametri
    n_epochs = 100
    BATCH_SIZE = 16
    EVAL_BATCH_SIZE = 300
    ROAD_LEN = 197  # Punti della strada
    TRAJ_LEN = 197  # Punti della traiettoria target

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print('Loading dataset... ', end='')
    with open('../data/sensodat.json') as f:
        roads, _, _, trajectories = load(f)
    print('Done.')

    roads, trajectories = select(roads, trajectories, where=lambda x: len(x[1])>11)


    def parse(road, traj):
        road = adjust_road_array(np.array(road, dtype=np.float32), target_size=TRAJ_LEN*4)
        traj = adjust_traj_array(np.array(traj, dtype=np.float32), target_size=TRAJ_LEN)
        traj = frenet_frame(traj, road)
        road = extract_road_features(adjust_road_array(road, ROAD_LEN))
        return road, traj


    print('Parsing data...', end='')
    roads, trajectories = eagle_feeder(parse, roads, trajectories, verbose=True)
    print('Done.')

    start_points = torch.zeros((EVAL_BATCH_SIZE, 2), dtype=torch.float32).to('cuda')
    start_points[:, 0], start_points[:, 1] = 2.5541313119361546, 2.5151659045117882



    # Normalizing
    roads_mean, roads_std = torch.mean(roads, dim=(0,1)), torch.std(roads, dim=(0,1))
    trajs_mean, trajs_std = torch.mean(trajectories, dim=(0,1)), torch.std(trajectories, dim=(0,1))
    print(f'means: {trajs_mean.tolist()}, stds: {trajs_std.tolist()}')

    roads = (roads - roads_mean) / roads_std
    trajectories = (trajectories - trajs_mean) / trajs_std

    start_points = (start_points - trajs_mean) / trajs_std



    print('Splitting train and test...', end='')
    (roads_train, roads_test,
     traj_train, traj_test,) = train_test_split(roads, trajectories, train_size=0.90, shuffle=True)
    print('Done.')

    print('Splitting train and valid...', end='')
    (roads_train, roads_valid,
     traj_train, traj_valid,) = train_test_split(roads_train, traj_train, train_size=0.89, shuffle=True)
    print('Done.')

    # Inizializza modello, loss e ottimizzatore
    model = TrajectoryTransformer(3, 2, 64, 4, 2).to(device)
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.9)


    stop_training = False
    def signal_handler(_, __):
        global stop_training
        stop_training = True
    signal.signal(signal.SIGINT, signal_handler)

    best_loss = float('inf')
    best_weights = deepcopy(model.state_dict())
    history = []
    loss, ade, fde = 0.0, 0.0, 0.0
    for epoch in range(n_epochs):
        for batch_start in range(0, len(roads_train), BATCH_SIZE):
            if stop_training:
                break
            # time_start = perf_counter()
            road, traj = roads_train[batch_start:batch_start+BATCH_SIZE], traj_train[batch_start:batch_start+BATCH_SIZE]
            loss, ade, fde = train_epoch(model, optimizer, criterion, road, traj, device)
            # ETA = int((perf_counter() - time_start) * (len(roads_valid) - batch_start - BATCH_SIZE) / 1e9)
            print(f"\rEpoch {epoch:2d} Training Loss:   ",
                  f"{loss:.4f} | ADE: {ade:.4f} | FDE: {fde:.4f}  ",
                  f"{batch_start:6}/{roads_train.shape[0]}  ",
                  # f"ETA: {human_time(ETA)}",
                  end='', sep='')
        if stop_training:
            break
        print( f"\rEpoch {epoch:2d} Training Loss:   ",
               f"{loss:.4f} | ADE: {ade:.4f} | FDE: {fde:.4f}  ",
               f"{roads_train.shape[0]:6}/{roads_train.shape[0]}",
               end='', sep='')


        metrics = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for batch_start in range(0, len(roads_valid), EVAL_BATCH_SIZE):
            if stop_training:
                break
            # time_start = perf_counter()
            roads = roads_valid[batch_start:batch_start+EVAL_BATCH_SIZE]
            trajs = traj_valid[batch_start:batch_start+EVAL_BATCH_SIZE]
            # start_points = trajs[:, 0, :]
            predicted_traj = predict_trajectory(model, roads, start_points[:roads.shape[0]], max_len=TRAJ_LEN - 1, device=device)
            ade, fde = compute_metrics(predicted_traj, trajs)
            loss = criterion(predicted_traj, trajs).item()
            metrics += np.array([loss, ade, fde, 1.0])
            # ETA = int((perf_counter() - time_start) * (len(roads_valid)-batch_start-EVAL_BATCH_SIZE) / 1e9)
            print(f"\rEpoch {epoch:2d} Validation Loss: ",
                  f"{loss:.4f} | ADE: {ade:.4f} | FDE: {fde:.4f}  ",
                  f"{batch_start:6}/{roads_valid.shape[0]}  ",
                  # f"ETA: {human_time(ETA)}",
                  end='', sep='')
        if stop_training:
            break
        loss, ade, fde = metrics[:3]/metrics[3]
        print(f"\rEpoch {epoch:2d} Validation Loss: ",
              f"{loss:.4f} | ADE: {ade:.4f} | FDE: {fde:.4f}  ",
              f"{roads_valid.shape[0]:6}/{roads_valid.shape[0]}",
              end='', sep='')

        # scheduler.step(loss)

        history.append(float(loss))
        if loss < best_loss:
            best_loss = loss
            best_weights = deepcopy(model.state_dict())
            print('  ***')
        else:
            print()

    model.load_state_dict(best_weights)


    metrics = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for batch_start in range(0, len(roads_test), EVAL_BATCH_SIZE):
        roads = roads_test[batch_start:batch_start + EVAL_BATCH_SIZE]
        trajs = traj_test[batch_start:batch_start + EVAL_BATCH_SIZE]
        # start_points = trajs[:, 0, :]
        predicted_traj = predict_trajectory(model, roads, start_points[:roads.shape[0]], max_len=TRAJ_LEN - 1, device=device)
        ade, fde = compute_metrics(predicted_traj, trajs)
        loss = criterion(predicted_traj, trajs).item()
        print(f"\rTest Loss:                {loss:.4f} | ADE: {ade:.4f} | FDE: {fde:.4f}  {batch_start:6}/{roads_test.shape[0]}", end='')
        metrics += np.array([loss, ade, fde, 1.0])

    loss, ade, fde = metrics[:3] / metrics[3]
    print(f"\rTest Loss:                {loss:.4f} | ADE: {ade:.4f} | FDE: {fde:.4f}  {roads_test.shape[0]:6}/{roads_test.shape[0]}")
    eagle_nester('../eagles/transf_eagle', model, '-tmp', history)
    plt.plot(history)
    plt.show()