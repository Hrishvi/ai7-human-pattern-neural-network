# human-pattern-nueral-network

> A from-scratch neural network that learns your personal number-picking patterns — no ML libraries required.

---

## What is this?

People aren't as random as they think. When asked to "pick a random number," humans follow subtle unconscious patterns — avoiding recent picks, favouring certain ranges, clustering around familiar values.

**pattern-prophet** trains a small neural network entirely from scratch (pure Python, zero dependencies) on *your* historical number sequences, then tries to predict what you'll pick next.

It's part curiosity, part experiment, and entirely built by hand — including forward passes, backpropagation, and weight updates.

---

## How it works

Training runs in two phases:

1. **Logistic Regression warm-up** — Quickly identifies which of your previous picks are most predictive of the next one. This gives the neural network a smart starting point rather than random noise.

2. **Neural Network training** — A two-hidden-layer network (configurable size) is seeded from the LR weights and trained with MSE loss + L2 regularisation. The output neuron is linear (no sigmoid), which suits the regression nature of the task.

```
Input (last N picks) → Hidden Layer 1 (sigmoid) → Hidden Layer 2 (sigmoid) → Output (linear)
```

Models are saved as `.pkl` files and can be retrained incrementally — new data merges with old.

---

## Quickstart

No installation required. Runs on Python 3.6+ with the standard library only.

### 1. Train a model

```bash
python train.py
```

- Enter your number sequences when prompted (space-separated, one sequence per line)
- Type `done` when finished
- Give the model a name — it saves to `Trained_Models/your_name.pkl`

### 2. Test / predict

```bash
python test.py
```

- Select your saved model
- Enter your recent picks
- The model predicts your next one

---

## Configuration

At the top of `train.py`, adjust these to match your use case:

```python
NUM_MIN  = 1        # lowest number in your range
NUM_MAX  = 10       # highest number in your range
WINDOW   = 10       # how many previous picks to use as input
```

Other tuneable hyperparameters:

| Parameter | Default | Effect |
|---|---|---|
| `hidden_neurons` | 20 | Size of first hidden layer |
| `hidden_neurons_2` | 12 | Size of second hidden layer |
| `learning_rate` | 0.01 | Step size during backprop |
| `epochs` | 100,000 | Training iterations |
| `l2_lambda` | 0.001 | Regularisation strength |

---

## Project structure

```
pattern-prophet/
├── train.py              # Collect sequences, train, save model
├── test.py               # Load model, enter picks, get prediction
├── Trained_Models/       # Saved .pkl model files (git-ignored)
└── README.md
```

---

## Honest disclaimer

Human number-picking patterns exist, but they're weak signals. This model will sometimes be right in an uncanny way — and often wrong. It's a fun experiment in what a neural network *can* learn from behavioural data, not a reliable oracle.

The real value here is educational: every neuron, weight update, and gradient is written by hand with no black-box libraries. Great for understanding how backpropagation actually works.

---

## Requirements

- Python 3.6+
- No third-party libraries (`math`, `random`, `pickle`, `os` only)

---

## Licence

MIT
