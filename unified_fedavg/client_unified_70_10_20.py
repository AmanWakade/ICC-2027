"""Five-client Flower DGF training; validation only during rounds.
Final held-out test: --test-checkpoint PATH --threshold FLOAT (offline, no Flower server).
Requires existing model.py and quant_codec.py in the same project directory.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import flwr as fl
from model import DGFMLP, get_weights, set_weights, set_seed
from quant_codec import encode, decode

DEFAULT_DATA = Path(r'D:\ICC conference\processed_70_10_20')


def load_data(path):
    with np.load(path, allow_pickle=False) as d:
        X = np.asarray(d['X'], dtype=np.float32)
        y = np.asarray(d['y'], dtype=np.int64)
    if X.ndim != 2 or X.shape[1] != 54 or X.shape[0] != len(y):
        raise ValueError(f'Unexpected shape: {path}: {X.shape}, {y.shape}')
    if not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
        raise ValueError(f'Invalid data: {path}')
    return X, y


def probabilities(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X)).detach().cpu().numpy().reshape(-1)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))


def metrics(y, p):
    eps = 1e-7
    pclip = np.clip(p, eps, 1 - eps)
    loss = float(-np.mean(y * np.log(pclip) + (1 - y) * np.log(1 - pclip)))
    return loss, {
        'roc_auc': float(roc_auc_score(y, p)),
        'pr_auc': float(average_precision_score(y, p)),
        'brier': float(brier_score_loss(y, p)),
    }


class DGFClient(fl.client.NumPyClient):
    def __init__(self, cid, data_dir, batch_size, lr, precision, seed, output_dir):
        self.cid, self.precision = cid, precision
        self.model = DGFMLP()
        self.lr = lr
        self.output_dir = output_dir
        self.Xtr, self.ytr = load_data(data_dir / f'client_{cid}_train.npz')
        self.Xval, self.yval = load_data(data_dir / f'client_{cid}_val.npz')
        self.train_loader = DataLoader(
            TensorDataset(torch.from_numpy(self.Xtr), torch.from_numpy(self.ytr.astype(np.float32))),
            batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed + cid),
        )
        print(f'Client {cid}: train={len(self.ytr)}, validation={len(self.yval)}', flush=True)

    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        epochs = int(config['local_epochs'])
        for _ in range(epochs):
            for xb, yb in self.train_loader:
                optimizer.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                optimizer.step()
        original = get_weights(self.model)
        if self.precision == 'fp32':
            compressed = [np.asarray(w, dtype=np.float32) for w in original]
            error = 0.0
        else:
            compressed = encode(original, self.precision)
            restored = decode(compressed, self.precision, [w.shape for w in original])
            a = np.concatenate([w.ravel().astype(np.float64) for w in original])
            b = np.concatenate([w.ravel().astype(np.float64) for w in restored])
            error = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))
        print(f'Client {self.cid}: fit round {config["server_round"]} completed', flush=True)
        return compressed, len(self.ytr), {'relative_quant_error': error}

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        p = probabilities(self.model, self.Xval)
        loss, scores = metrics(self.yval, p)
        # Local validation predictions are saved for post-training threshold selection.
        # They are NOT sent as patient-level data over Flower.
        dest = self.output_dir / 'validation_predictions'
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest / f'client_{self.cid}_round_{int(config["server_round"]):03d}.npz',
                            y_true=self.yval.astype(np.uint8), y_prob=p.astype(np.float32))
        print(f'Client {self.cid}: validation round {config["server_round"]}, loss={loss:.5f}', flush=True)
        return loss, len(self.yval), scores


def test_checkpoint(args):
    X, y = load_data(args.data_dir / f'client_{args.client_id}_test.npz')
    with np.load(args.test_checkpoint, allow_pickle=False) as d:
        arrays = [d[f'arr_{i}'] for i in range(len(d.files))]
    model = DGFMLP()
    set_weights(model, arrays)
    p = probabilities(model, X)
    loss, scores = metrics(y, p)
    pred = p >= args.threshold
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    scores.update({'loss': loss, 'threshold': args.threshold, 'tp': tp, 'tn': tn,
                   'fp': fp, 'fn': fn, 'sensitivity': tp / max(tp + fn, 1),
                   'specificity': tn / max(tn + fp, 1)})
    dest = args.output_dir / 'test_predictions'
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / f'client_{args.client_id}_test_predictions.npz',
                        y_true=y.astype(np.uint8), y_prob=p.astype(np.float32))
    import json
    (dest / f'client_{args.client_id}_metrics.json').write_text(json.dumps(scores, indent=2))
    print(f'Client {args.client_id} TEST (best validation checkpoint): {scores}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--precision', choices=['fp32', 'fp16', 'int8'], required=True)
    ap.add_argument('--client-id', type=int, required=True, choices=range(1, 6))
    ap.add_argument('--server', default='127.0.0.1:8080')
    ap.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--test-checkpoint', type=Path, default=None)
    ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args()
    set_seed(args.seed + args.client_id)
    if args.test_checkpoint is not None:
        if not 0 <= args.threshold <= 1:
            ap.error('--threshold must be between 0 and 1')
        test_checkpoint(args)
    else:
        client = DGFClient(args.client_id, args.data_dir, args.batch_size,
                           args.lr, args.precision, args.seed, args.output_dir)
        fl.client.start_numpy_client(server_address=args.server, client=client)


if __name__ == '__main__':
    main()
