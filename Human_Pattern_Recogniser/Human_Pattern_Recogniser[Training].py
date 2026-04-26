"""
==========================
Trains a two-hidden-layer neural network (built entirely from scratch)
on your personal number sequences, then saves the model to disk.

Training runs in two phases:
  Phase 1 — Logistic Regression: quickly finds which past picks matter most.
  Phase 2 — Neural Network:      seeded from LR weights, trained with
                                  backpropagation and MSE + L2 loss.

No third-party libraries are used. Requires Python 3.6+.
"""

import random
import math
import pickle
import os


# =============================================
# CONFIG
# Adjust these to match your number game/range.
# =============================================

NUM_MIN  = 1       # The lowest number you ever pick
NUM_MAX  = 10      # The highest number you ever pick
WINDOW   = 10      # How many past picks are used as inputs to the model

n_inputs = WINDOW  # Input layer size equals the window (one node per past pick)

# Hidden layer sizes — larger = more capacity, slower training
hidden_neurons   = 20   # Neurons in hidden layer 1
hidden_neurons_2 = 12   # Neurons in hidden layer 2

# Training hyperparameters
learning_rate = 0.01     # How large each gradient step is
epochs        = 100_000  # Total passes over the training data
l2_lambda     = 0.001    # L2 regularisation strength (penalises large weights to reduce overfitting)


# =============================================
# HELPERS
# =============================================

def sigmoid(x):
    """
    Sigmoid activation function: maps any real number to (0, 1).
    Used in hidden layers to introduce non-linearity.
    Clamps input to [-500, 500] to prevent math overflow.
    """
    x = max(-500.0, min(500.0, x))
    return 1 / (1 + math.exp(-x))


def normalize(value):
    """
    Maps a number from [NUM_MIN, NUM_MAX] → [0.0, 1.0].
    Neural networks train much better when inputs are on a common small scale.
    """
    return (value - NUM_MIN) / (NUM_MAX - NUM_MIN)


def denormalize(value):
    """
    Inverse of normalize: maps a model output [0.0, 1.0] → [NUM_MIN, NUM_MAX].
    Used to convert the network's raw output back into a human-readable pick.
    """
    return round(value * (NUM_MAX - NUM_MIN) + NUM_MIN)


def he_init(fan_in):
    """
    He (Kaiming) weight initialisation: samples from a Gaussian with
    std = sqrt(2 / fan_in). Designed for layers followed by ReLU/sigmoid —
    helps prevent vanishing/exploding gradients at the start of training.
    """
    return random.gauss(0, math.sqrt(2.0 / fan_in))


def linear(x):
    """
    Identity / linear activation — returns x unchanged.
    Used on the output neuron because this is a regression task:
    we want a continuous number, not a probability capped at 1.
    """
    return x


# =============================================
# COLLECT TRAINING DATA FROM THE USER
# The user enters sequences of past picks.
# More data = better learned patterns.
# =============================================

print("=" * 55)
print("  Human Pattern Trainer — Personal Number Predictor")
print("=" * 55)
print(f"\n  Range  : {NUM_MIN} – {NUM_MAX}")
print(f"  Window : {WINDOW} previous picks → predict next\n")
print("  Enter your number sequences below.")
print("  Type numbers separated by spaces, e.g:  7 3 45 12 88 ...")
print("  Press ENTER after each sequence.")
print("  Type 'done' when finished.\n")

all_numbers = []  # Master list of every number entered across all sequences

while True:
    line = input("  Sequence: ").strip()

    if line.lower() == "done":
        break

    if not line:
        continue  # Skip blank lines silently

    try:
        nums = [int(x) for x in line.split()]
        # Clamp each number to the valid range in case the user typed out-of-bounds values
        nums = [max(NUM_MIN, min(NUM_MAX, n)) for n in nums]
        all_numbers.extend(nums)
        print(f"  ✓ Added {len(nums)} numbers  (total so far: {len(all_numbers)})\n")
    except ValueError:
        print("  ✗ Could not parse that line — use spaces between numbers.\n")

# Need at least WINDOW + 1 numbers to form even one training sample
if len(all_numbers) <= WINDOW:
    print(f"\n  Need more than {WINDOW} numbers to train. Quitting.")
    exit()

print(f"\n  Total numbers collected : {len(all_numbers)}")


# =============================================
# BUILD TRAINING DATA WITH A SLIDING WINDOW
#
# For each position i in all_numbers, we take:
#   inputs = the WINDOW numbers before position i  (normalised)
#   target = the number at position i              (normalised)
#
# Example with WINDOW=3 and sequence [4, 7, 2, 5, 1]:
#   [4, 7, 2] → predict 5
#   [7, 2, 5] → predict 1
# =============================================

training_data = []

for i in range(len(all_numbers) - WINDOW):
    window = all_numbers[i : i + WINDOW]   # Slice of WINDOW picks (context)
    target = all_numbers[i + WINDOW]        # The very next pick (label)

    inputs      = [normalize(v) for v in window]
    norm_target = normalize(target)

    # Store as a single flat list: [input_0, input_1, ..., input_n, target]
    # The trailing element is always the target; everything before it is inputs.
    training_data.append(inputs + [norm_target])

print(f"  Training samples built  : {len(training_data)}")
print(f"  Architecture            : {n_inputs} → {hidden_neurons} → {hidden_neurons_2} → 1\n")


# =============================================
# PHASE 1 — LOGISTIC REGRESSION WARM-UP
#
# Before the full neural network, we train a simple logistic regression
# model on the same data. Its purpose is two-fold:
#   1. Quickly identify which window positions (e.g. "pick -1", "pick -3")
#      are actually predictive of the next number.
#   2. Provide warm-start weights for hidden layer 1 so the neural network
#      begins with a meaningful signal rather than pure noise.
#
# Logistic regression uses sigmoid output and MSE loss here (unusual but
# pragmatic since we're using it as a regression seed, not a classifier).
# =============================================

print("=" * 55)
print("Phase 1: Logistic Regression (warm-up)")
print("=" * 55)

# Initialise weights with small random values near zero
lr_weights = [random.uniform(-0.1, 0.1) for _ in range(n_inputs)]
lr_bias    = 0.0
lr_rate    = 0.05  # Slightly higher learning rate for faster Phase 1 convergence

for epoch in range(20_000):
    total_loss = 0

    for *features, target in training_data:
        # Forward pass: weighted sum → sigmoid
        z      = sum(features[j] * lr_weights[j] for j in range(n_inputs)) + lr_bias
        output = sigmoid(z)

        # MSE loss contribution from this sample
        error       = target - output
        total_loss += error ** 2

        # Gradient of sigmoid MSE: error * sigmoid_derivative
        # sigmoid'(z) = output * (1 - output)
        grad = error * output * (1 - output)

        # Update each weight and the bias using gradient ascent on error
        for j in range(n_inputs):
            lr_weights[j] += lr_rate * grad * features[j]
        lr_bias += lr_rate * grad

    if epoch % 5000 == 0:
        print(f"  LR Epoch {epoch:>6} | Loss: {total_loss / len(training_data):.6f}")

# Print which window positions the LR found most informative
print(f"\n  Top weighted positions (1 = most recent pick):")
indexed = sorted(enumerate(lr_weights), key=lambda x: abs(x[1]), reverse=True)
for rank, (pos, w) in enumerate(indexed[:5], 1):
    label = f"pick -{WINDOW - pos}"
    print(f"    #{rank}  {label:>8}  weight: {w:+.4f}")


# =============================================
# PHASE 2 — NEURAL NETWORK TRAINING
#
# Architecture:
#   Input (WINDOW nodes) → Hidden 1 (sigmoid) → Hidden 2 (sigmoid) → Output (linear)
#
# Key design choices:
#   - Hidden layer 1 seeded from LR weights (warm start)
#   - Linear output neuron → network can predict any real value, not just [0,1]
#   - L2 regularisation → penalises large weights, reduces overfitting
#   - Best-weights tracking → restores the checkpoint with lowest loss
# =============================================

print("\n" + "=" * 55)
print("Phase 2: Neural Network Training")
print("=" * 55)

# ── Weight initialisation ─────────────────────────────────────────────────────

# Hidden layer 1: seed each neuron's weights from the LR weights, plus small Gaussian noise.
# This gives the network a head start — it already "knows" which positions matter.
hidden_layer_1 = [
    [lr_weights[j] + random.gauss(0, 0.05) for j in range(n_inputs)]
    for _ in range(hidden_neurons)
]

# Hidden layer 2 and output: He initialisation (good default for sigmoid layers)
hidden_layer_2 = [[he_init(hidden_neurons)  for _ in range(hidden_neurons)] for _ in range(hidden_neurons_2)]
output_neuron  = [he_init(hidden_neurons_2) for _ in range(hidden_neurons_2)]

# Biases: seed HL1 biases from the LR bias (+ noise); others start at zero
bias_hidden_1  = [lr_bias + random.gauss(0, 0.01) for _ in range(hidden_neurons)]
bias_hidden_2  = [0.0] * hidden_neurons_2
bias_output    = 0.0

# Track the best weights seen so far across all epochs
best_loss    = float('inf')
best_weights = None


# ── Training loop ─────────────────────────────────────────────────────────────

for epoch in range(epochs):

    random.shuffle(training_data)
    total_loss = 0

    for *features, target in training_data:

        inputs = features  # Normalised picks [0.0 – 1.0], length = WINDOW

        # ── Forward pass ──────────────────────────────────────────────────────

        # Hidden layer 1: each neuron computes a weighted sum of all inputs,
        # adds its bias, then passes through sigmoid.
        h1 = []
        for i in range(hidden_neurons):
            s = sum(inputs[j] * hidden_layer_1[i][j] for j in range(n_inputs)) + bias_hidden_1[i]
            h1.append(sigmoid(s))

        # Hidden layer 2: same structure, receives h1 activations as input.
        h2 = []
        for i in range(hidden_neurons_2):
            s = sum(h1[j] * hidden_layer_2[i][j] for j in range(hidden_neurons)) + bias_hidden_2[i]
            h2.append(sigmoid(s))

        # Output neuron: linear (no sigmoid). The weighted sum IS the prediction.
        # This lets the network predict values outside [0,1] if needed, and avoids
        # squashing that would hurt regression accuracy near the extremes.
        output_sum = sum(h2[i] * output_neuron[i] for i in range(hidden_neurons_2)) + bias_output
        output     = linear(output_sum)

        # ── Loss: MSE + L2 regularisation ─────────────────────────────────────

        error = target - output   # Residual (positive = under-predicted, negative = over)

        # L2 penalty: sum of squared weights (biases excluded by convention).
        # Adding this to the loss discourages any individual weight from growing
        # very large, which is a common cause of overfitting.
        l2_penalty = (
            sum(w**2 for row in hidden_layer_1 for w in row) +
            sum(w**2 for row in hidden_layer_2 for w in row) +
            sum(w**2 for w in output_neuron)
        )
        total_loss += error**2 + l2_lambda * l2_penalty

        # ── Backpropagation ───────────────────────────────────────────────────
        #
        # We propagate the error gradient backwards through the network,
        # layer by layer, using the chain rule of calculus.
        #
        # For a linear output: d(loss)/d(output) = -2 * error
        # But since we're doing gradient *ascent* on error (error = target - output),
        # the effective output gradient is just `error` (sign absorbed by update rule).

        # Output layer gradient
        # Linear activation: derivative is 1, so gradient passes straight through.
        output_gradient    = error
        old_output_weights = output_neuron[:]  # Snapshot before update (needed for HL2 backprop)

        for i in range(hidden_neurons_2):
            # Weight update: += lr * (gradient contribution from h2[i]) - L2 decay term
            output_neuron[i] += learning_rate * (output_gradient * h2[i] - l2_lambda * output_neuron[i])
        bias_output += learning_rate * output_gradient  # Bias has no L2 penalty

        # Hidden layer 2 gradients
        # For sigmoid: derivative = h2[i] * (1 - h2[i])
        # We need to snapshot HL2 weights before updating them so HL1 backprop
        # uses the correct (pre-update) values.
        old_hl2  = [row[:] for row in hidden_layer_2]
        grads_h2 = []

        for i in range(hidden_neurons_2):
            # Error flowing back into neuron i of HL2:
            # = output_gradient * weight connecting h2[i] to output
            hidden_error    = output_gradient * old_output_weights[i]
            # Multiply by sigmoid derivative to get the gradient at h2[i]
            hidden_gradient = hidden_error * h2[i] * (1 - h2[i])
            grads_h2.append(hidden_gradient)

            for j in range(hidden_neurons):
                hidden_layer_2[i][j] += learning_rate * (hidden_gradient * h1[j] - l2_lambda * hidden_layer_2[i][j])
            bias_hidden_2[i] += learning_rate * hidden_gradient

        # Hidden layer 1 gradients
        # Error flowing back into neuron i of HL1 = sum over all HL2 neurons of
        # (grad_h2[k] * weight from h1[i] to h2[k])
        for i in range(hidden_neurons):
            hidden_error    = sum(grads_h2[k] * old_hl2[k][i] for k in range(hidden_neurons_2))
            hidden_gradient = hidden_error * h1[i] * (1 - h1[i])

            for j in range(n_inputs):
                hidden_layer_1[i][j] += learning_rate * (hidden_gradient * inputs[j] - l2_lambda * hidden_layer_1[i][j])
            bias_hidden_1[i] += learning_rate * hidden_gradient

    # ── Track best weights ────────────────────────────────────────────────────

    avg_loss = total_loss / len(training_data)

    if avg_loss < best_loss:
        best_loss = avg_loss
        # Deep-copy all weight matrices so we can restore this checkpoint later
        best_weights = {
            "hidden_layer_1": [row[:] for row in hidden_layer_1],
            "hidden_layer_2": [row[:] for row in hidden_layer_2],
            "output_neuron":  output_neuron[:],
            "bias_hidden_1":  bias_hidden_1[:],
            "bias_hidden_2":  bias_hidden_2[:],
            "bias_output":    bias_output,
        }

    if epoch % 10000 == 0:
        print(f"  Epoch {epoch:>7} | Loss: {avg_loss:.6f} | Best: {best_loss:.6f}")

print(f"\n  Training complete. Best loss: {best_loss:.6f}")


# =============================================
# RESTORE BEST WEIGHTS
# Replace current weights with the checkpoint
# that achieved the lowest loss during training.
# =============================================

hidden_layer_1 = best_weights["hidden_layer_1"]
hidden_layer_2 = best_weights["hidden_layer_2"]
output_neuron  = best_weights["output_neuron"]
bias_hidden_1  = best_weights["bias_hidden_1"]
bias_hidden_2  = best_weights["bias_hidden_2"]
bias_output    = best_weights["bias_output"]


# =============================================
# SAVE MODEL TO DISK
#
# The entire model state is serialised into a
# .pkl file inside the Trained_Models/ folder.
#
# If a model with the same name already exists,
# the new training numbers are MERGED with the
# old ones so the model accumulates data over
# multiple training sessions.
# =============================================

save_dir = "Trained_Models"
os.makedirs(save_dir, exist_ok=True)  # Create folder if it doesn't exist

model_name = input("\nEnter model name: ").strip()
if not model_name:
    model_name = "human_pattern"  # Default name if user pressed Enter

save_path = os.path.join(save_dir, model_name + ".pkl")

# Merge with any existing model data so history accumulates across sessions
existing_numbers = []
if os.path.exists(save_path):
    with open(save_path, "rb") as f:
        old = pickle.load(f)
    existing_numbers = old.get("all_numbers", [])
    print(f"  Merging with existing model ({len(existing_numbers)} previous numbers).")

merged_numbers = existing_numbers + all_numbers

# Package all model components into a single dictionary for easy retrieval
model_data = {
    "NUM_MIN":          NUM_MIN,           # Number range (needed to de-normalise predictions)
    "NUM_MAX":          NUM_MAX,
    "WINDOW":           WINDOW,            # Input size the model expects
    "n_inputs":         n_inputs,
    "hidden_neurons":   hidden_neurons,    # Architecture metadata
    "hidden_neurons_2": hidden_neurons_2,
    "hidden_layer_1":   hidden_layer_1,   # Weight matrices
    "hidden_layer_2":   hidden_layer_2,
    "output_neuron":    output_neuron,
    "bias_hidden_1":    bias_hidden_1,    # Bias vectors
    "bias_hidden_2":    bias_hidden_2,
    "bias_output":      bias_output,
    "lr_weights":       lr_weights,       # Logistic regression weights (used in test.py for baseline)
    "lr_bias":          lr_bias,
    "all_numbers":      merged_numbers,   # Full number history for display / future merges
}

with open(save_path, "wb") as f:
    pickle.dump(model_data, f)

print(f"  Model saved to: {save_path}")
print(f"  Total numbers stored in model: {len(merged_numbers)}")
