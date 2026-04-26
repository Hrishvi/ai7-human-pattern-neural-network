import pickle
import math
import sys
import os


# =============================================
# HELPERS
# =============================================

def sigmoid(x):
    x = max(-500.0, min(500.0, x))
    return 1 / (1 + math.exp(-x))

def normalize(value, num_min, num_max):
    return (value - num_min) / (num_max - num_min)

def denormalize(value, num_min, num_max):
    return value * (num_max - num_min) + num_min

def clamp(value, lo, hi):
    return max(lo, min(hi, value))

def predict(inputs, model):
    n_inputs       = model["n_inputs"]
    hidden_neurons   = model["hidden_neurons"]
    hidden_neurons_2 = model["hidden_neurons_2"]
    hl1  = model["hidden_layer_1"]
    hl2  = model["hidden_layer_2"]
    out  = model["output_neuron"]
    bh1  = model["bias_hidden_1"]
    bh2  = model["bias_hidden_2"]
    bo   = model["bias_output"]

    h1 = [sigmoid(sum(inputs[j] * hl1[i][j] for j in range(n_inputs)) + bh1[i])
          for i in range(hidden_neurons)]

    h2 = [sigmoid(sum(h1[j] * hl2[i][j] for j in range(hidden_neurons)) + bh2[i])
          for i in range(hidden_neurons_2)]

    raw = sum(h2[i] * out[i] for i in range(hidden_neurons_2)) + bo
    return raw  # linear output

def confidence_label(spread):
    """Give a confidence label based on prediction spread."""
    if spread <= 5:
        return "high"
    elif spread <= 15:
        return "medium"
    else:
        return "low"


# =============================================
# LOAD MODEL
# =============================================

save_dir = "Trained_Models"

if not os.path.exists(save_dir):
    print(f"No '{save_dir}' folder found. Train a model first.")
    sys.exit()

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

with open(load_path, "rb") as f:
    model = pickle.load(f)

NUM_MIN = model["NUM_MIN"]
NUM_MAX = model["NUM_MAX"]
WINDOW  = model["WINDOW"]

print(f"\nModel loaded : {model_name}")
print(f"Range        : {NUM_MIN} – {NUM_MAX}")
print(f"Window       : last {WINDOW} picks")
print(f"Trained on   : {len(model.get('all_numbers', []))} numbers")

# Show which past positions the LR found most influential
lr_weights = model["lr_weights"]
indexed    = sorted(enumerate(lr_weights), key=lambda x: abs(x[1]), reverse=True)
print(f"\nYour top pattern positions (most predictive):")
for rank, (pos, w) in enumerate(indexed[:3], 1):
    label = f"pick -{WINDOW - pos}"
    print(f"  #{rank}  {label}  (weight: {w:+.4f})")


# =============================================
# SESSION HISTORY
# Keeps track of what you've entered this
# session so you don't have to retype picks.
# =============================================

session_history = []

print("\n" + "=" * 50)
print("  Testing — enter your last picks one by one")
print("  The model predicts what you'll pick next.")
print("=" * 50)


# =============================================
# TESTING LOOP
# =============================================

while True:
    answer = input("\nDo you want to test the network? [y/n]: ").strip().lower()

    if answer != "y":
        print("Program ending.")
        break

    # ── Collect last WINDOW picks ─────────────────────────────────
    if len(session_history) >= WINDOW:
        # Auto-use session history, just ask for the most recent
        print(f"\n  Using your last {WINDOW} picks from this session:")
        window = session_history[-WINDOW:]
        print(f"  {window}")
        new_pick = input(f"\n  Enter your most recent pick ({NUM_MIN}–{NUM_MAX}): ").strip()
        try:
            new_pick = int(new_pick)
            new_pick = int(clamp(new_pick, NUM_MIN, NUM_MAX))
            session_history.append(new_pick)
            window = session_history[-WINDOW:]
        except ValueError:
            print("  Invalid input.")
            continue
    else:
        # Need to collect WINDOW picks manually
        needed = WINDOW - len(session_history)
        print(f"\n  Enter your last {needed} pick(s) (oldest first), {NUM_MIN}–{NUM_MAX}:")
        new_picks = []
        for i in range(needed):
            while True:
                try:
                    val = int(input(f"    Pick {len(session_history) + i + 1}: "))
                    val = int(clamp(val, NUM_MIN, NUM_MAX))
                    new_picks.append(val)
                    break
                except ValueError:
                    print("  Please enter a valid number.")
        session_history.extend(new_picks)
        window = session_history[-WINDOW:]

    # ── Normalize window ──────────────────────────────────────────
    inputs = [normalize(v, NUM_MIN, NUM_MAX) for v in window]

    # ── Neural network prediction ─────────────────────────────────
    raw_output  = predict(inputs, model)
    nn_pred_raw = denormalize(raw_output, NUM_MIN, NUM_MAX)
    nn_pred     = round(clamp(nn_pred_raw, NUM_MIN, NUM_MAX))

    # ── Logistic regression prediction (simpler baseline) ─────────
    lr_weights_m = model["lr_weights"]
    lr_bias_m    = model["lr_bias"]
    lr_z         = sum(inputs[j] * lr_weights_m[j] for j in range(WINDOW)) + lr_bias_m
    lr_out       = sigmoid(lr_z)
    lr_pred      = round(clamp(denormalize(lr_out, NUM_MIN, NUM_MAX), NUM_MIN, NUM_MAX))

    # ── Confidence: based on distance from midpoint ───────────────
    mid    = (NUM_MIN + NUM_MAX) / 2
    spread = abs(nn_pred - mid)
    conf   = confidence_label(spread)

    # ── Top 3 candidates around the prediction ────────────────────
    candidates = sorted(
        set(clamp(nn_pred + offset, NUM_MIN, NUM_MAX) for offset in range(-2, 3)),
        key=lambda x: abs(x - nn_pred_raw)
    )[:3]

    # ── Display ───────────────────────────────────────────────────
    print(f"\n  Your last {WINDOW} picks : {window}")
    print(f"\n  Neural Network predicts : {nn_pred}  (confidence: {conf})")
    print(f"  Logistic Regression     : {lr_pred}")
    print(f"  Top candidates          : {candidates}")