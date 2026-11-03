"""Flower FedAvg with FP16 or symmetric INT8 uplink quantisation.
FP32 downlink, FP32 server aggregation; 5 clients, 20 rounds, 20 local epochs.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from model import DGFMLP, get_weights, set_seed
from quant_codec import decode


def aggregate_eval(metrics):
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics)
    keys = ['roc_auc', 'pr_auc', 'brier', 'sensitivity_05', 'specificity_05',
            'f1_05', 'precision_05', 'sensitivity_03', 'specificity_03',
            'f1_03', 'precision_03']
    out = {k: float(sum(n * m[k] for n, m in metrics) / total) for k in keys}
    out['worst_centre_sensitivity_05'] = min(m['sensitivity_05'] for _, m in metrics)
    out['worst_centre_sensitivity_03'] = min(m['sensitivity_03'] for _, m in metrics)
    return out


class QuantFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, precision, output_dir, rounds, reference_shapes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.precision = precision
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rounds = rounds
        self.reference_shapes = reference_shapes
        self.rows = []
        self.bytes_by_round = {}
        self.quant_error_by_round = {}

    def aggregate_fit(self, server_round, results, failures):
        if failures or len(results) != 5:
            raise RuntimeError(f'Round {server_round}: expected five successful clients; got {len(results)}')
        uplink_bytes = 0
        weighted_error = 0.0
        weighted_count = 0
        for _, fit_res in results:
            weighted_error += fit_res.num_examples * float(fit_res.metrics.get('relative_quant_error', 0.0))
            weighted_count += fit_res.num_examples
            # Count actual serialized ndarray payloads (NumPy .npy headers included),
            # excluding gRPC/protobuf/TLS transport overhead.
            uplink_bytes += sum(len(t) for t in fit_res.parameters.tensors)
            compressed = parameters_to_ndarrays(fit_res.parameters)
            restored = decode(compressed, self.precision, self.reference_shapes)
            fit_res.parameters = ndarrays_to_parameters(restored)
        self.bytes_by_round[server_round] = uplink_bytes
        self.quant_error_by_round[server_round] = weighted_error / weighted_count if weighted_count else 0.0
        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            arrays = parameters_to_ndarrays(aggregated)
            np.savez_compressed(self.output_dir / 'latest_global_fp32.npz', *arrays)
            if server_round == self.rounds:
                np.savez_compressed(self.output_dir / 'final_global_fp32.npz', *arrays)
        print(f'Round {server_round} uplink tensor bytes: {uplink_bytes:,}', flush=True)
        return aggregated, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        if failures or len(results) != 5:
            raise RuntimeError(f'Round {server_round}: incomplete evaluation')
        if metrics:
            row = {'round': server_round, 'weighted_loss': loss,
                   'uplink_tensor_bytes': self.bytes_by_round.get(server_round, 0),
                   'mean_relative_quant_error': self.quant_error_by_round.get(server_round, 0.0), **metrics}
            self.rows.append(row)
            with (self.output_dir / 'round_metrics.csv').open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(self.rows)
            print(f'Round {server_round}: {row}', flush=True)
        return loss, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--precision', required=True, choices=['fp32', 'fp16', 'int8'])
    ap.add_argument('--address', default='127.0.0.1:8080')
    ap.add_argument('--rounds', type=int, default=20)
    ap.add_argument('--local-epochs', type=int, default=5)
    ap.add_argument('--output-dir', type=Path, default=None)
    args = ap.parse_args()
    out = args.output_dir or Path(__file__).resolve().parent / 'results' / f'fedavg_{args.precision}'
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'Results folder is not empty: {out}. Move/archive it or specify --output-dir.')
    set_seed(42)
    initial_arrays = get_weights(DGFMLP())
    strategy = QuantFedAvg(
        precision=args.precision, output_dir=out, rounds=args.rounds,
        reference_shapes=[a.shape for a in initial_arrays],
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=5, min_evaluate_clients=5, min_available_clients=5,
        initial_parameters=ndarrays_to_parameters(initial_arrays),
        on_fit_config_fn=lambda r: {'server_round': r, 'local_epochs': args.local_epochs},
        on_evaluate_config_fn=lambda r: {'server_round': r, 'total_rounds': args.rounds},
        evaluate_metrics_aggregation_fn=aggregate_eval,
        accept_failures=False,
    )
    import json
    with (out / 'experiment_config.json').open('w', encoding='utf-8') as f:
        json.dump({'precision': args.precision, 'rounds': args.rounds, 'local_epochs': args.local_epochs,
                   'clients': 5, 'seed_server': 42, 'seed_client_rule': '42 + client_id',
                   'model': 'MLP 54-64-32-1', 'optimizer': 'Adam', 'learning_rate': 0.001,
                   'batch_size': 128, 'downlink': 'fp32', 'uplink': args.precision,
                   'scaling': 'existing client-local scalers (not harmonised)',
                   'thresholds': [0.3, 0.5]}, f, indent=2)
    print(f'Configuration: {args.rounds} rounds, {args.local_epochs} local epochs; results: {out}', flush=True)
    print(f'Waiting for five {args.precision.upper()} clients...', flush=True)
    fl.server.start_server(server_address=args.address,
                           config=fl.server.ServerConfig(num_rounds=args.rounds), strategy=strategy)

if __name__ == '__main__':
    main()
