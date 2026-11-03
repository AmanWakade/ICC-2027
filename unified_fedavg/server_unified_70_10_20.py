"""Unified FP32/FP16/INT8 Flower FedAvg; best checkpoint chosen by validation log loss.
Only client->server weights are quantised. Server aggregation and downlink are FP32.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from model import DGFMLP, get_weights, set_seed
from quant_codec import decode


def aggregate_eval(items):
    if not items:
        return {}
    total = sum(n for n, _ in items)
    return {key: float(sum(n * m[key] for n, m in items) / total)
            for key in ('roc_auc', 'pr_auc', 'brier')}


class QuantFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, precision, output_dir, rounds, reference_shapes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.precision, self.output_dir, self.rounds = precision, output_dir, rounds
        self.reference_shapes = reference_shapes
        self.rows, self.bytes_by_round, self.quant_error_by_round = [], {}, {}
        self.best_loss, self.best_round = float('inf'), None
        self.latest_arrays = None

    def aggregate_fit(self, server_round, results, failures):
        if failures or len(results) != 5:
            raise RuntimeError(f'Round {server_round}: expected five successful clients, got {len(results)}')
        uplink_bytes, weighted_error, count = 0, 0.0, 0
        for _, fit_res in results:
            uplink_bytes += sum(len(t) for t in fit_res.parameters.tensors)
            weighted_error += fit_res.num_examples * float(fit_res.metrics['relative_quant_error'])
            count += fit_res.num_examples
            encoded = parameters_to_ndarrays(fit_res.parameters)
            print(
               f"Round {server_round}: received {len(encoded)} tensors, "
               f"dtypes={[str(a.dtype) for a in encoded]}",
               flush=True
                )
            restored = ([np.asarray(a, dtype=np.float32) for a in encoded]
                        if self.precision == 'fp32'
                        else decode(encoded, self.precision, self.reference_shapes))
            if len(restored) != len(self.reference_shapes):
                raise ValueError(
                   f"Round {server_round}: parameter count mismatch: "
                   f"{len(restored)} vs {len(self.reference_shapes)}"
                )

            for j, (a, shape) in enumerate(zip(restored, self.reference_shapes)):
                if a.shape != shape:
                   raise ValueError(
                       f"Round {server_round}: tensor {j} shape mismatch: "
                       f"{a.shape} vs {shape}"
                    )
                if not np.isfinite(a).all():
                   raise ValueError(
                        f"Round {server_round}: tensor {j} contains "
                        f"{np.isnan(a).sum()} NaN and {np.isinf(a).sum()} Inf values"
                    )
            fit_res.parameters = ndarrays_to_parameters(restored)
        self.bytes_by_round[server_round] = uplink_bytes
        self.quant_error_by_round[server_round] = weighted_error / count
        aggregated, fit_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated is None:
            raise RuntimeError('Missing aggregated parameters')
        self.latest_arrays = parameters_to_ndarrays(aggregated)
        np.savez_compressed(self.output_dir / 'latest_global_fp32.npz', *self.latest_arrays)
        if server_round == self.rounds:
            np.savez_compressed(self.output_dir / 'final_round_global_fp32.npz', *self.latest_arrays)
        print(f'Round {server_round}: uplink tensor bytes={uplink_bytes:,}', flush=True)
        return aggregated, fit_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        if failures or len(results) != 5:
            raise RuntimeError(f'Round {server_round}: incomplete validation')
        loss, scores = super().aggregate_evaluate(server_round, results, failures)
        if loss is None or not np.isfinite(loss):
            raise RuntimeError(f'Round {server_round}: invalid validation loss')
        row = {'round': server_round, 'validation_loss': float(loss),
               'uplink_tensor_bytes': self.bytes_by_round[server_round],
               'mean_relative_quant_error': self.quant_error_by_round[server_round], **scores}
        self.rows.append(row)
        with (self.output_dir / 'round_metrics.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(self.rows)
        if loss < self.best_loss:
            self.best_loss, self.best_round = float(loss), server_round
            np.savez_compressed(self.output_dir / 'best_global_fp32.npz', *self.latest_arrays)
            (self.output_dir / 'best_checkpoint.json').write_text(json.dumps({
                'best_round': server_round, 'validation_loss': self.best_loss,
                'selection_metric': 'sample-weighted validation log loss',
            }, indent=2), encoding='utf-8')
            print(f'New best checkpoint: round {server_round}, val loss={loss:.6f}', flush=True)
        return loss, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--precision', required=True, choices=['fp32', 'fp16', 'int8'])
    ap.add_argument('--address', default='127.0.0.1:8080')
    ap.add_argument('--rounds', type=int, default=20)
    ap.add_argument('--local-epochs', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f'Results folder not empty: {args.output_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    initial_arrays = get_weights(DGFMLP())
    strategy = QuantFedAvg(
        precision=args.precision, output_dir=args.output_dir, rounds=args.rounds,
        reference_shapes=[a.shape for a in initial_arrays],
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=5, min_evaluate_clients=5, min_available_clients=5,
        initial_parameters=ndarrays_to_parameters(initial_arrays),
        on_fit_config_fn=lambda r: {'server_round': r, 'local_epochs': args.local_epochs},
        on_evaluate_config_fn=lambda r: {'server_round': r},
        evaluate_metrics_aggregation_fn=aggregate_eval,
        accept_failures=False,
    )
    (args.output_dir / 'experiment_config.json').write_text(json.dumps({
        'precision': args.precision, 'rounds': args.rounds, 'local_epochs': args.local_epochs,
        'clients': 5, 'seed_server': args.seed, 'seed_client_rule': 'seed + client_id',
        'model': 'MLP 54-64-32-1', 'optimizer': 'Adam', 'learning_rate': 0.001,
        'batch_size': 128, 'downlink': 'fp32', 'uplink': args.precision,
        'preprocessing': 'shared train-only fitted preprocessing; pooled statistics assumed available',
        'checkpoint_selection': 'lowest weighted validation log loss',
        'test_evaluation': 'offline after checkpoint and threshold selection',
        'communication_accounting': 'serialized uplink ndarray tensors only, not full network traffic',
    }, indent=2), encoding='utf-8')
    print(f'{args.precision.upper()}: {args.rounds} rounds x {args.local_epochs} local epochs; seed={args.seed}', flush=True)
    print('Waiting for five clients...', flush=True)
    fl.server.start_server(server_address=args.address,
                           config=fl.server.ServerConfig(num_rounds=args.rounds), strategy=strategy)
    print(f'Finished. Best checkpoint: {args.output_dir / "best_global_fp32.npz"}', flush=True)


if __name__ == '__main__':
    main()
