"""Run five terminals: python client.py --client-id 1 (through 5)."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss, f1_score, balanced_accuracy_score, precision_score, recall_score
import flwr as fl
from model import DGFMLP, get_weights, set_weights, set_seed
from quant_codec import encode, decode

DEFAULT_DATA = Path(r'D:\ICC conference\processed')


def load_data(path):
    # allow_pickle is only appropriate for these locally generated, trusted NPZ files.
    with np.load(path, allow_pickle=True) as d:
        X = np.asarray(d['X'], dtype=np.float32)
        y = np.asarray(d['y'], dtype=np.int64)
    if X.ndim != 2 or X.shape[1] != 54 or len(X) != len(y):
        raise ValueError(f'Unexpected dataset shape: {path}: {X.shape}, {y.shape}')
    if not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
        raise ValueError(f'Invalid values in {path}')
    return X, y


def classification_metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        'sensitivity': float(recall_score(y, pred, zero_division=0)),
        'specificity': float(tn / (tn + fp)) if tn + fp else float('nan'),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


class DGFClient(fl.client.NumPyClient):
    def __init__(self, cid, data_dir, batch_size, lr, precision):
        self.precision = precision
        self.cid = cid
        self.model = DGFMLP()
        self.batch_size = batch_size
        self.lr = lr
        self.Xtr, self.ytr = load_data(data_dir / f'client_{cid}_train.npz')
        self.Xte, self.yte = load_data(data_dir / f'client_{cid}_test.npz')
        self.train_loader = DataLoader(
            TensorDataset(torch.from_numpy(self.Xtr), torch.from_numpy(self.ytr.astype(np.float32))),
            batch_size=batch_size, shuffle=True,
        )
        print(f'Client {cid}: train={len(self.ytr)}, test={len(self.yte)}, positives={self.yte.sum()}', flush=True)

    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        epochs = int(config.get('local_epochs', 20))
        total_loss = 0.0
        for _ in range(epochs):
            for xb, yb in self.train_loader:
                optimizer.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
        print(f'Client {self.cid}: completed round {config.get("server_round", "?")}', flush=True)
        original = get_weights(self.model)
        compressed = encode(original, self.precision)
        restored = decode(compressed, self.precision, [w.shape for w in original])
        flat_original = np.concatenate([w.ravel().astype(np.float64) for w in original])
        flat_restored = np.concatenate([w.ravel().astype(np.float64) for w in restored])
        error = float(np.linalg.norm(flat_original - flat_restored) / max(np.linalg.norm(flat_original), 1e-12))
        return compressed, len(self.ytr), {'train_loss_batch_sum': total_loss, 'relative_quant_error': error}

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(self.Xte)).numpy()
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
        eps = 1e-7
        loss = float(-np.mean(self.yte * np.log(np.clip(p, eps, 1-eps)) + (1-self.yte) * np.log(np.clip(1-p, eps, 1-eps))))
        m05 = classification_metrics(self.yte, p, .5)
        m03 = classification_metrics(self.yte, p, .3)
        metrics = {
            'roc_auc': float(roc_auc_score(self.yte, p)) if len(np.unique(self.yte)) > 1 else 0.0,
            'pr_auc': float(average_precision_score(self.yte, p)) if len(np.unique(self.yte)) > 1 else 0.0,
            'brier': float(brier_score_loss(self.yte, p)),
            'sensitivity_05': m05['sensitivity'], 'specificity_05': m05['specificity'],
            'f1_05': m05['f1'], 'precision_05': m05['precision'],
            'sensitivity_03': m03['sensitivity'], 'specificity_03': m03['specificity'],
            'f1_03': m03['f1'], 'precision_03': m03['precision'],
        }
        # Save held-out predictions locally only at the final round.
        # This is for post-hoc plots; never send patient-level predictions to the server.
        if int(config.get('server_round', -1)) == int(config.get('total_rounds', 20)):
            target = Path(__file__).resolve().parent / 'results' / f'fedavg_{self.precision}' / 'predictions'
            target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target / f'client_{self.cid}_test_predictions.npz',
                                y_true=self.yte.astype(np.uint8), y_prob=p.astype(np.float32))
            print(f'Client {self.cid}: saved held-out predictions to {target}', flush=True)
        print(f'Client {self.cid} round {config.get("server_round", "?")}: ROC-AUC={metrics["roc_auc"]:.4f} PR-AUC={metrics["pr_auc"]:.4f} sens@0.3={metrics["sensitivity_03"]:.4f}', flush=True)
        return loss, len(self.yte), metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--precision', choices=['fp32', 'fp16', 'int8'], required=True)
    ap.add_argument('--client-id', type=int, required=True, choices=range(1, 6))
    ap.add_argument('--server', default='127.0.0.1:8080')
    ap.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.001)
    args = ap.parse_args()
    set_seed(42 + args.client_id)
    client = DGFClient(args.client_id, args.data_dir, args.batch_size, args.lr, args.precision)
    fl.client.start_numpy_client(server_address=args.server, client=client)

if __name__ == '__main__':
    main()
