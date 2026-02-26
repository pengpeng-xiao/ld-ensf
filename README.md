# LD-EnSF: Latent Dynamics with Ensemble Score Filters

Code for the paper: *Xiao, P., Si, P., and Chen, P. LD-EnSF: Synergizing latent dynamics with ensemble score filters for fast data assimilation with sparse observations. The Fourteenth International Conference on Learning Representations (ICLR), 2026.*

## Overview

LD-EnSF combines latent-dynamics neural networks with ensemble score filters to perform fast data assimilation with sparse observations across three domains:
- Atmospheric modeling (Shallow Water Equations on sphere)
- Tsunami modeling (Shallow Water Equations)
- Kolmogorov flow (2D turbulent fluid dynamics)

## Installation

### Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies:
- PyTorch >= 1.10.0
- DeepXDE >= 1.10.0
- NumPy >= 1.20.0
- wandb (for experiment tracking)

Optional (for data generation):
- JAX >= 0.3.0
- jax-cfd >= 0.1.0

## Repository Structure

```
.
├── src/                        # Core library
│   ├── model.py               # LD-EnSF model architectures
│   ├── train.py               # Training utilities
│   ├── encoder.py             # Encoder networks (CNN, LSTM, RNN)
│   ├── dataloader.py          # Data loading utilities
│   ├── normalization.py       # Normalization classes
│   └── utils.py               # Helper functions
├── kolmogorov_flow/           # Kolmogorov flow experiments
│   ├── train_ldnet.py         # Train latent dynamics network
│   ├── train_lstm.py          # Train LSTM encoder
│   ├── observation_data.py    # Generate observation data
│   ├── ldensf_assimilation.py # Run data assimilation
│   └── data_generation.py     # Generate training data (JAX)
├── atmospheric_modeling/      # Atmospheric modeling experiments
│   └── [similar structure]
└── tsunami_modeling/          # Tsunami modeling experiments
    └── [similar structure]
```

## Usage

### Running from Repository Root

Set PYTHONPATH to the repository root:

```bash
export PYTHONPATH=/path/to/ld-ensf:$PYTHONPATH
```

### Training a Model

Each domain has training scripts that require a `--base-path` argument:

```bash
# Kolmogorov flow example
cd kolmogorov_flow
python train_ldnet.py --base-path /path/to/data/kolmogorov_flow
```

Key arguments:
- `--base-path`: Base directory containing data and models (required)
- `--data-path`: Relative path to training data
- `--model-path`: Relative path for saving checkpoints
- `--num-latent-states`: Dimension of latent space
- `--device`: CUDA device (e.g., "cuda:0")

See individual training scripts for full argument lists.

### Data Assimilation

After training, run data assimilation:

```bash
python ldensf_assimilation.py --base-path /path/to/data/kolmogorov_flow
```

## Model Architecture

### Latent Dynamics Network (LDNN)

- **Dynamics network**: Maps latent state + parameters → latent state derivative
- **Reconstruction network**: Maps latent state + space → physical state
- Supports Fourier embeddings and residual connections

### Variants

- `LDNN`: Basic latent dynamics model
- `ResLDNN`: With residual connections in dynamics
- `FourierLDNN`: With Fourier feature embeddings
- `ResFourierLDNN`: Combines both enhancements

## Citation

If you use this code for academic research, please cite:

```bibtex
@inproceedings{xiao2026ldensf,
  title={{LD}-{E}n{SF}: Synergizing latent dynamics with ensemble score filters for fast data assimilation with sparse observations},
  author={Xiao, Pengpeng and Si, Peng and Chen, Peng},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=XXXXXXXX}
}
```

## Questions

Open an issue in the GitHub "Issues" section for help with the code or data.

## License

MIT License
