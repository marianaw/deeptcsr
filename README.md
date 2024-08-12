# survan

## Non-exhaustive list of required packages

- `tqdm`
- `yaml`
- `jax` (for our experiments we used version `0.4.25`)
- `optax` 
- `haiku`
- `dataclasses`

## Basic usage

To run experiments on the PBC dataset run

```bash
python main.py --agent DeepLambdaSA --config configs/config_pbc.yaml
```

where `--agent` can be `SA` for the Survival Analysis baseline, `LambdaSA` for our implementation of the algorithm by Maystre and Russo and `DeepLambdaSA` for our algorithm.

Use the configuration file to define hyperparameters such as the architecture backbone, the lambda value, whether or not to use landmarking, etc.
Results are stored under the folder specified by the `output_file` entry in the configuration file (see `config_pbc.yaml` for an example). To see Concordance Index and Integrated Brier Score metrics look for the `results.json` located under a path of the form `<algorithm>/<dataset>/<backbone>/<lambda>/<landmark>/<exp_name>/<seed>`.

