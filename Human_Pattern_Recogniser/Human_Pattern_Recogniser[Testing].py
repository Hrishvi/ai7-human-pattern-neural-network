"""
=========================
Loads a trained model from disk and runs an interactive prediction loop.

For each round, the user provides their last N picks (where N = the
WINDOW size the model was trained on). The script returns:
  - A neural network prediction
  - A logistic regression baseline prediction
  - A confidence label
  - The top 3 candidate numbers near the prediction

No third-party libraries are used. Requires Python 3.6+.
Run train.py first to create a model.
"""

import pickle
import math
import sys
import os


# =============================================
# HELPERS
# =============================================

def sigmoid(x):
    """
    Sigmoid activation: maps any real number to (0, 1).
    Clamps input to avoid overflow in math.exp for very large/small values.
    Must match the sigmoid used during training exactly.
    """
    x = max(-500.0, min(500.0, x))
    return 1 / (1 + math.exp(-x))


def normalize(value, num_min, num_max):
    """
    Maps a number from [num_min, num_max] → [0.0, 1.0].
    The model was trained on normalised inputs, so test inputs must be
    normalised the same way before being fed in.
    """
    return (value - num_min) / (num_max - num_min)


def denormalize(value, num_min, num_max):
    """
    Inverse of normalize: maps a value from [0.0, 1.0] → [num_min, num_max].
    Used to convert the network's raw output back to a pick in the user's range.
    """
    return value * (num_max - num_min) + num_min


def clamp(value, lo, hi):
    """
    Clips a value to stay within [lo, hi].
    Prevents predictions from straying outside the valid number range.
    """
    return max(lo, min(hi, value))


def predict(inputs, model):
    """
    Runs a full forward pass through the neural network.

    Parameters
    ----------
    inputs : list of float
        Normalised window of past picks, length = model["WINDOW"].
    model  : dict
        The loaded model dictionary containing all weights and biases.

    Returns
    -------
    float
        Raw (linear) output of the network. This is still normalised [~0, ~1]
        and needs to be denormalised before being shown to the user.
    """
    # Unpack architecture parameters from the model dictionary
    n_inputs         = model["n_inputs"]
    hidden_neurons   = model["hidden_neurons"]
    hidden_neurons_2 = model["hidden_neurons_2"]
    hl1  = model["hidden_layer_1"]   # Weights: hidden layer 1
    hl2  = model["hidden_layer_2"]   # Weights: hidden layer 2
    out  = model["output_neuron"]    # Weights: output neuron
    bh1  = model["bias_hidden_1"]
    bh2  = model["bias_hidden_2"]
    bo   = model["bias_output"]

    # Hidden layer 1: weighted sum of inputs + bias → sigmoid
    h1 = [
        sigmoid(sum(inputs[j] * hl1[i][j] for j in range(n_inputs)) + bh1[i])
        for i in range(hidden_neurons)
    ]

    # Hidden layer 2: weighted sum of h1 activations + bias → sigmoid
    h2 = [
        sigmoid(sum(h1[j] * hl2[i][j] for j in range(hidden_neurons)) + bh2[i])
        for i in range(hidden_neurons_2)
    ]

    # Output neuron: linear (no activation). Returns raw weighted sum.
    # A linear output is used because this is regression, not classification.
    raw = sum(h2[i] * out[i] for i in range(hidden_neurons_2)) + bo
    return raw


def confidence_label(spread):
    """
    Converts a distance-from-midpoint value into a human-readable confidence label.

    A prediction far from the midpoint of the range is considered more confident
    because the network is committing to one end of the scale rather than hedging
    near the centre.

    Parameters
    ----------
    spread : float
        Absolute distance between the predicted number and the midpoint of the range.

    Returns
    -------
    str
        "high", "medium", or "low"
    """
    if spread <= 5:
        return "high"
    elif spread <= 15:
        return "medium"
    else:
        return "low"


# =============================================
# LOAD MODEL
# Lists all available saved models and lets
# the user choose which one to load.
# =============================================

save_dir = "Trained_Models"

if not os.path.exists(save_dir):
    print(f"No '{save_dir}' folder found. Train a model first.")
    sys.exit()

# Only list files that end in .pkl (our model format)
available = [f for f in os.listdir(save_dir) if f.endswith(".pkl")]

if not available:
    print(f"No saved models found in '{save_dir}'. Train a model first.")
    sys.exit()

print("Available models:")
for name in available:
    print(f"  - {name.replace('.pkl', '')}")

model_name = input("\nEnter the model name to load: ").strip()

if not model_name:
    print("No name entered. Quitting.")
    sys.exit()

load_path = os.path.join(save_dir, model_name + ".pkl")

if not os.path.exists(load_path):
    print(f"Model '{model_name}' not found. Quitting.")
    sys.exit()

# Deserialise the model dictionary from disk
with open(load_path, "rb") as f:
    model = pickle.load(f)

# Extract the config values the model was trained with.
# These must be used consistently during testing — e.g. normalisation
# must use the same NUM_MIN and NUM_MAX as training.
NUM_MIN = model["NUM_MIN"]
NUM_MAX = model["NUM_MAX"]
WINDOW  = model["WINDOW"]

# Summary display
print(f"\nModel loaded : {model_name}")
print(f"Range        : {NUM_MIN} – {NUM_MAX}")
print(f"Window       : last {WINDOW} picks")
print(f"Trained on   : {len(model.get('all_numbers', []))} numbers")

# Show which past pick positions the logistic regression found most predictive.
# A large absolute weight → that position strongly influences the next pick.
lr_weights = model["lr_weights"]
indexed    = sorted(enumerate(lr_weights), key=lambda x: abs(x[1]), reverse=True)
print(f"\nYour top pattern positions (most predictive):")
for rank, (pos, w) in enumerate(indexed[:3], 1):
    # pos=0 means the oldest pick in the window; pos=WINDOW-1 means the most recent.
    # "pick -N" = N picks ago.
    label = f"pick -{WINDOW - pos}"
    print(f"  #{rank}  {label}  (weight: {w:+.4f})")


# =============================================
# SESSION HISTORY
#
# Keeps a running list of every pick entered
# this session. Once enough picks accumulate,
# the script auto-fills the window rather than
# asking the user to retype previous picks
# every round.
# =============================================

session_history = []

print("\n" + "=" * 50)
print("  Testing — enter your last picks one by one")
print("  The model predicts what you'll pick next.")
print("=" * 50)


# =============================================
# PREDICTION LOOP
# Repeats until the user types "n".
# =============================================

while True:
    answer = input("\nDo you want to test the network? [y/n]: ").strip().lower()

    if answer != "y":
        print("Program ending.")
        break

    # ── Collect the last WINDOW picks ────────────────────────────────────────

    if len(session_history) >= WINDOW:
        # We already have enough history from this session.
        # Show the last WINDOW picks so the user can confirm context,
        # then only ask for the single newest pick.
        print(f"\n  Using your last {WINDOW} picks from this session:")
        window = session_history[-WINDOW:]
        print(f"  {window}")

        new_pick = input(f"\n  Enter your most recent pick ({NUM_MIN}–{NUM_MAX}): ").strip()
        try:
            new_pick = int(new_pick)
            new_pick = int(clamp(new_pick, NUM_MIN, NUM_MAX))  # Enforce valid range
            session_history.append(new_pick)
            window = session_history[-WINDOW:]  # Re-slice to get the updated window
        except ValueError:
            print("  Invalid input.")
            continue

    else:
        # Not enough history yet — collect the remaining picks manually.
        # "needed" = how many more picks are required to fill a full window.
        needed = WINDOW - len(session_history)
        print(f"\n  Enter your last {needed} pick(s) (oldest first), {NUM_MIN}–{NUM_MAX}:")

        new_picks = []
        for i in range(needed):
            while True:
                try:
                    val = int(input(f"    Pick {len(session_history) + i + 1}: "))
                    val = int(clamp(val, NUM_MIN, NUM_MAX))
                    new_picks.append(val)
                    break  # Move on to the next pick once a valid int is entered
                except ValueError:
                    print("  Please enter a valid number.")

        session_history.extend(new_picks)
        window = session_history[-WINDOW:]

    # ── Normalise the input window ────────────────────────────────────────────
    # Convert each pick from raw integer → [0.0, 1.0] before feeding the model.
    inputs = [normalize(v, NUM_MIN, NUM_MAX) for v in window]

    # ── Neural network prediction ─────────────────────────────────────────────
    raw_output  = predict(inputs, model)                          # Raw normalised output
    nn_pred_raw = denormalize(raw_output, NUM_MIN, NUM_MAX)       # Convert to pick scale
    nn_pred     = round(clamp(nn_pred_raw, NUM_MIN, NUM_MAX))     # Round and clamp to valid range

    # ── Logistic regression baseline prediction ───────────────────────────────
    # The simpler LR model serves as a baseline to compare the NN against.
    # It uses the same inputs but a single-layer sigmoid mapping.
    lr_weights_m = model["lr_weights"]
    lr_bias_m    = model["lr_bias"]
    lr_z         = sum(inputs[j] * lr_weights_m[j] for j in range(WINDOW)) + lr_bias_m
    lr_out       = sigmoid(lr_z)   # LR output is [0.0, 1.0] (sigmoid)
    lr_pred      = round(clamp(denormalize(lr_out, NUM_MIN, NUM_MAX), NUM_MIN, NUM_MAX))

    # ── Confidence label ──────────────────────────────────────────────────────
    # A prediction committed to one end of the range is considered more confident
    # than one that hovers near the midpoint.
    mid    = (NUM_MIN + NUM_MAX) / 2
    spread = abs(nn_pred - mid)
    conf   = confidence_label(spread)

    # ── Top 3 candidates ─────────────────────────────────────────────────────
    # Produce a small set of nearby numbers sorted by closeness to the raw
    # (un-rounded) prediction. Useful when the user wants to hedge their guess.
    candidates = sorted(
        # Generate offsets -2 to +2 around the predicted pick, clamped to valid range
        set(clamp(nn_pred + offset, NUM_MIN, NUM_MAX) for offset in range(-2, 3)),
        key=lambda x: abs(x - nn_pred_raw)  # Sort by distance from the raw (continuous) prediction
    )[:3]  # Keep only the 3 closest

    # ── Display results ───────────────────────────────────────────────────────
    print(f"\n  Your last {WINDOW} picks : {window}")
    print(f"\n  Neural Network predicts : {nn_pred}  (confidence: {conf})")
    print(f"  Logistic Regression     : {lr_pred}")
    print(f"  Top candidates          : {candidates}")
