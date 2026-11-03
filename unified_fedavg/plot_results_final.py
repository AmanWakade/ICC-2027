"""Plot FP32/FP16/INT8 Flower DGF results without retraining.

Place this file as plot_results.py inside unified_fedavg and run:
    python plot_results.py

Requires round_metrics.csv and final_global_fp32.npz in each final results folder,
and the original processed/client_{1..5}_test.npz datasets.
Predictions are generated from saved weights; no retraining is performed.
"""
from pathlib import Path
import csv
import numpy as np
import torch
from model import DGFMLP, set_weights
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score,
    average_precision_score, confusion_matrix,
)

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT / 'results'
DATA_DIR = PROJECT.parent / 'processed'
MODES = ('fp32', 'fp16', 'int8')
FOLDERS = {
    'fp32': 'fedavg_fp32_final',
    'fp16': 'fedavg_fp16_final',
    'int8': 'fedavg_int8_final',
}
EXPECTED_ROUNDS = 20
THRESHOLD = 0.3  # Exploratory threshold, not validation-selected.
OUT = ROOT / 'comparison'
OUT.mkdir(parents=True, exist_ok=True)


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=300, bbox_inches='tight')
    plt.close()


def get_base(mode):
    base = ROOT / FOLDERS[mode]
    if not base.is_dir():
        raise FileNotFoundError(f'{mode}: Missing results folder: {base}')
    return base


def read_rounds(mode):
    path = get_base(mode) / 'round_metrics.csv'
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    numbers = [int(r['round']) for r in rows]
    if numbers != list(range(1, EXPECTED_ROUNDS + 1)):
        raise ValueError(f'{mode}: Expected rounds 1–{EXPECTED_ROUNDS}, got {numbers}')
    return rows


def read_predictions(mode):
    base = get_base(mode)
    weights_path = base / 'final_global_fp32.npz'
    if not weights_path.is_file():
        raise FileNotFoundError(f'{mode}: Missing final model: {weights_path}')
    with np.load(weights_path, allow_pickle=False) as d:
        keys = sorted(d.files, key=lambda k: int(k.split('_')[-1]))
        weights = [np.asarray(d[k], dtype=np.float32) for k in keys]
    model = DGFMLP()
    set_weights(model, weights)
    model.eval()
    pred_dir = base / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)
    centre_data = {}
    for cid in range(1, 6):
        data_path = DATA_DIR / f'client_{cid}_test.npz'
        if not data_path.is_file():
            raise FileNotFoundError(f'Missing original test data: {data_path}')
        # The original trusted NPZ datasets may include object-typed metadata.
        with np.load(data_path, allow_pickle=True) as d:
            X = np.asarray(d['X'], dtype=np.float32)
            y = np.asarray(d['y'], dtype=np.int64).reshape(-1)
        if X.ndim != 2 or X.shape[1] != 54 or len(X) != len(y):
            raise ValueError(f'Unexpected test data dimensions: {data_path}')
        if not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
            raise ValueError(f'Invalid test data: {data_path}')
        with torch.no_grad():
            logits = model(torch.from_numpy(X)).cpu().numpy()
        p = (1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))).astype(np.float64)
        path = pred_dir / f'client_{cid}_test_predictions.npz'
        np.savez_compressed(path, y_true=y, y_prob=p)
        centre_data[cid] = (y, p)
    y_all = np.concatenate([centre_data[c][0] for c in range(1, 6)])
    p_all = np.concatenate([centre_data[c][1] for c in range(1, 6)])
    return centre_data, y_all, p_all


def write_csv(path, columns, records):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows(records)


def metrics_row(mode, cid, y, p):
    tn, fp, fn, tp = confusion_matrix(y, p >= THRESHOLD, labels=[0, 1]).ravel()
    roc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float('nan')
    ap = average_precision_score(y, p) if len(np.unique(y)) == 2 else float('nan')
    sensitivity = tp / (tp + fn) if (tp + fn) else float('nan')
    specificity = tn / (tn + fp) if (tn + fp) else float('nan')
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return [mode, cid, len(y), int(y.sum()), roc, ap,
            sensitivity, specificity, precision, f1,
            int(tp), int(fp), int(tn), int(fn)]


def draw_cm(cm, title, filename):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1], ['No DGF', 'DGF'])
    ax.set_yticks([0, 1], ['No DGF', 'DGF'])
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(title)
    cutoff = cm.max() / 2 if cm.size else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                    color='white' if cm[i, j] > cutoff else 'black')
    save_plot(filename)


def main():
    rounds = {mode: read_rounds(mode) for mode in MODES}
    predictions = {mode: read_predictions(mode) for mode in MODES}

    for metric, label in [
        ('roc_auc', 'Centre-weighted ROC-AUC'),
        ('pr_auc', 'Centre-weighted PR-AUC'),
        ('brier', 'Centre-weighted Brier score'),
        ('sensitivity_03', 'Centre-weighted sensitivity @ 0.3'),
        ('weighted_loss', 'Centre-weighted evaluation loss'),
    ]:
        plt.figure(figsize=(7, 4.5))
        for mode in MODES:
            rows = rounds[mode]
            plt.plot([int(r['round']) for r in rows],
                     [float(r[metric]) for r in rows], label=mode.upper())
        plt.xlabel('Federated round')
        plt.ylabel(label)
        plt.xticks(range(1, EXPECTED_ROUNDS + 1, 2))
        plt.legend()
        plt.grid(alpha=.2)
        save_plot(f'{metric}_by_round.png')

    plt.figure(figsize=(6, 5))
    for mode in MODES:
        _, y, p = predictions[mode]
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f'{mode.upper()} (AUC {roc_auc_score(y, p):.3f})')
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Chance')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    save_plot('pooled_roc_curves.png')

    plt.figure(figsize=(6, 5))
    for mode in MODES:
        _, y, p = predictions[mode]
        precision, recall, _ = precision_recall_curve(y, p)
        plt.plot(recall, precision,
                 label=f'{mode.upper()} (AP {average_precision_score(y, p):.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    save_plot('pooled_pr_curves.png')

    counts = {mode: sum(int(r['uplink_tensor_bytes']) for r in rounds[mode]) for mode in MODES}
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar([m.upper() for m in MODES], [counts[m] for m in MODES])
    plt.ylabel(f'Total serialized uplink tensor bytes ({EXPECTED_ROUNDS} rounds)')
    for bar, mode in zip(bars, MODES):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{counts[mode]:,}', ha='center', va='bottom', fontsize=9)
    plt.ylim(0, max(counts.values()) * 1.14)
    save_plot('total_uplink_bytes.png')

    centre_rows = []
    pooled_rows = []
    for mode in MODES:
        centre_data, y_all, p_all = predictions[mode]
        for cid, (y, p) in centre_data.items():
            centre_rows.append(metrics_row(mode, cid, y, p))
            cm = confusion_matrix(y, p >= THRESHOLD, labels=[0, 1])
            draw_cm(cm, f'{mode.upper()} – centre {cid} @ {THRESHOLD}',
                    f'confusion_{mode}_centre_{cid}.png')
        cm = confusion_matrix(y_all, p_all >= THRESHOLD, labels=[0, 1])
        draw_cm(cm, f'{mode.upper()} – pooled @ {THRESHOLD}',
                f'confusion_{mode}_pooled.png')
        pooled_rows.append([
            mode, len(y_all), roc_auc_score(y_all, p_all),
            average_precision_score(y_all, p_all), counts[mode],
            100 * (1 - counts[mode] / counts['fp32']),
        ])

    write_csv(OUT / 'centre_metrics.csv',
              ['precision', 'client_id', 'n', 'positives', 'roc_auc', 'pr_auc',
               'sensitivity_03', 'specificity_03', 'precision_03', 'f1_03',
               'tp', 'fp', 'tn', 'fn'], centre_rows)
    write_csv(OUT / 'pooled_summary.csv',
              ['precision', 'n_test', 'pooled_roc_auc', 'pooled_pr_auc',
               'uplink_tensor_bytes', 'uplink_reduction_percent'], pooled_rows)
    print(f'Saved corrected figures, confusion matrices, and CSVs in: {OUT}')
    print('Note: communication counts exclude downlink and transport overhead.')
    print('Note: threshold 0.3 is exploratory; not validation-selected.')


if __name__ == '__main__':
    main()
