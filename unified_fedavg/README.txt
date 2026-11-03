UNIFIED FEDAVG PATCH (existing codebase, not a new project)

1. Back up existing unified_fedavg/results first. Existing experiment folders must be empty before starting each run.
2. Replace ONLY unified_fedavg/server.py and unified_fedavg/client.py with these two files.
   Keep your existing model.py and quant_codec.py unchanged.
3. Copy plot_results.py into the same unified_fedavg directory.
4. Run server: python -u server.py --precision fp32
   In FIVE other terminals run:
   python -u client.py --client-id 1 --precision fp32
   ... similarly client-id 2,3,4,5.
5. Repeat separately with --precision fp16 and then --precision int8.
   Defaults are 20 federated rounds and 5 local epochs. Never use fp8 for INT8.
6. After all 3 modes finish, run: python plot_results.py

Results: unified_fedavg/results/fedavg_fp32, fedavg_fp16, fedavg_int8
Charts: unified_fedavg/results/comparison
Each mode saves 20-round metrics, model weights, quantization error and local per-client
held-out predictions. plot_results.py produces pooled ROC/PR curves, centre metrics,
communication and convergence figures.

IMPORTANT:
- These are held-out TEST predictions. Do not tune thresholds on them.
- Patient-level prediction files are stored locally in this single-machine simulation.
  They are not uploaded to the Flower server. Real multi-site deployments require
  appropriate permissions for pooling predictions.
- Results are a single seed, NOT a statistical confidence interval.
- Existing client-local feature scalers are not harmonized.
- --output-dir custom paths are not supported for the prediction-saving client;
  use default results paths for this workflow.
