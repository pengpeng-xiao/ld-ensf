# LD-EnSF: Latent Dynamics with Ensemble Score Filters

Code for the paper: *Xiao, P., Si, P., and Chen, P. LD-EnSF: Synergizing latent dynamics with ensemble score filters for fast data assimilation with sparse observations. The Fourteenth International Conference on Learning Representations (ICLR), 2026.*

## Installation

```bash
pip install -r requirements.txt
```

Core dependencies:
- PyTorch >= 1.10.0
- DeepXDE >= 1.10.0
- NumPy >= 1.20.0
- wandb (for experiment tracking)

Optional dependencies for data generation:
- Kolmogorov flow: JAX >= 0.3.0, jax-cfd >= 0.1.0
- Atmospheric modeling: Dedalus >= 3.0 

## Repository Structure

```
.
├── src/                            # Core library
│   ├── model.py                   # Model architectures (LDNN, ResLDNN, FourierLDNN, ResFourierLDNN)
│   ├── train.py / trainer.py      # Training utilities
│   ├── encoder.py                 # Encoder networks (CNN, LSTM, RNN)
│   ├── dataloader.py              # Data loading
│   ├── data_preprocess.py         # Preprocessing
│   ├── normalization.py           # Normalization classes
│   └── utils.py                   # Helper functions
├── kolmogorov_flow/               # Kolmogorov flow experiments
├── atmospheric_modeling/          # Atmospheric modeling experiments
└── tsunami_modeling/              # Tsunami modeling experiments
```

Each domain folder contains the same five scripts: `data_generation.py`, `train_ldnet.py`, `observation_data.py`, `train_lstm.py`, `ldensf_assimilation.py`.

## Pipeline

The workflow follows five steps. Examples below use Kolmogorov flow; other domains are identical unless noted. Set `PYTHONPATH` to the repository root before running:

```bash
export PYTHONPATH=/path/to/ld-ensf:$PYTHONPATH
```

### Step 1 — Generate Data

```bash
python kolmogorov_flow/data_generation.py
```

> **Atmospheric modeling** uses a Dedalus-based MPI solver instead. `data_generation.sh` runs the solver across 200 trajectories with randomized forcing parameters and splits the output into 120 train / 40 valid / 40 test directories:
> ```bash
> bash atmospheric_modeling/data_generation.sh
> ```

### Step 2 — Train the Latent Dynamics Network

```bash
python kolmogorov_flow/train_ldnet.py \
    --base-path /path/to/base \
    --data-path kolmogorov_flow/data/kolmogorov_data.npz \
    --model-path kolmogorov_flow/saved_model/ldnet \
    --device cuda:0
```

### Step 3 — Generate Observation Data with Latent Trajectories

Run the trained LDNet to extract latent state trajectories and add sparse observation points. Output is saved as a `.pth` file used in Step 4.

```bash
python kolmogorov_flow/observation_data.py \
    --base-path /path/to/base \
    --data-path kolmogorov_flow/data/kolmogorov_data.npz \
    --model-path kolmogorov_flow/saved_model/ldnet
```

### Step 4 — Train the LSTM Encoder

Train an LSTM to map sparse observation sequences to latent trajectories.

```bash
python kolmogorov_flow/train_lstm.py \
    --data-path kolmogorov_flow/data/observation_data.pth \
    --save-path kolmogorov_flow/saved_model/lstm
```

### Step 5 — Offline Data Assimilation (LD-EnSF)

```bash
python kolmogorov_flow/ldensf_assimilation.py \
    --base-path /path/to/base \
    --data-path kolmogorov_flow/data/observation_data.pth \
    --model-path kolmogorov_flow/saved_model/ldnet
```

## Citation

```bibtex
@inproceedings{
  xiao2026ldensf,
  title={{LD}-En{SF}: Synergizing Latent Dynamics with Ensemble Score Filters for Fast Data Assimilation with Sparse Observations},
  author={Pengpeng Xiao and Phillip Si and Peng Chen},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=AWSVzzhbr7}
}
```

## Questions

Open an issue in the GitHub "Issues" section for help with the code or data.
