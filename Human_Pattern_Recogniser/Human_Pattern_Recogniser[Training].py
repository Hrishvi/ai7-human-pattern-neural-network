import random
import math
import pickle
import os

# =============================================
# CONFIG — set your number range here
# =============================================

NUM_MIN    = 1      # lowest number you pick
NUM_MAX    = 10   # highest number you pick  ← change to your range
WINDOW     = 10     # how many previous picks to look back at
n_inputs   = WINDOW

hidden_neurons   = 20
hidden_neurons_2 = 12

learning_rate = 0.01
epochs        = 100_000
l2_lambda     = 0.001


# =============================================
# HELPERS
# =============================================

def sigmoid(x):
    x = max(-500.0, min(500.0, x))
    return 1 / (1 + math.exp(-x))

def normalize(value):
    """Map a number in [NUM_MIN, NUM_MAX] to [0.0, 1.0]"""
    return (value - NUM_MIN) / (NUM_MAX - NUM_MIN)

def denormalize(value):
    """Map a model output [0.0, 1.0] back to [NUM_MIN, NUM_MAX]"""
    return round(value * (NUM_MAX - NUM_MIN) + NUM_MIN)

def he_init(fan_in):
    return random.gauss(0, math.sqrt(2.0 / fan_in))

def linear(x):
    """Linear activation for output neuron — suits regression tasks."""
    return x


# =============================================
# INPUT YOUR NUMBER SEQUENCES
# Type numbers separated by spaces.
# You can enter multiple sequences one at a time.
# The more numbers you provide, the better.
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

all_numbers = []

while True:
    line = input("  Sequence: ").strip()
    if line.lower() == "done":
        break
    if not line:
        continue
    try:
        nums = [int(x) for x in line.split()]
        # Clamp to valid range
        nums = [max(NUM_MIN, min(NUM_MAX, n)) for n in nums]
        all_numbers.extend(nums)
        print(f"  ✓ Added {len(nums)} numbers  (total so far: {len(all_numbers)})\n")
    except ValueError:
        print("  ✗ Could not parse that line — use spaces between numbers.\n")

if len(all_numbers) <= WINDOW:
    print(f"\n  Need more than {WINDOW} numbers to train. Quitting.")
    exit()

print(f"\n  Total numbers collected : {len(all_numbers)}")


# =============================================
# BUILD TRAINING DATA FROM SLIDING WINDOW
# Each window of WINDOW picks → predict the next
# =============================================

training_data = []

for i in range(len(all_numbers) - WINDOW):
    window = all_numbers[i : i + WINDOW]
    target = all_numbers[i + WINDOW]

    inputs      = [normalize(v) for v in window]
    norm_target = normalize(target)
    training_data.append(inputs + [norm_target])

print(f"  Training samples built  : {len(training_data)}")
print(f"  Architecture            : {n_inputs} → {hidden_neurons} → {hidden_neurons_2} → 1\n")


# =============================================
# PHASE 1 — LOGISTIC REGRESSION (warm-up)
# Learns which of the 10 previous picks matter
# most for predicting the next one.
# =============================================

print("=" * 55)
print("Phase 1: Logistic Regression (warm-up)")
print("=" * 55)

lr_weights = [random.uniform(-0.1, 0.1) for _ in range(n_inputs)]
lr_bias    = 0.0
lr_rate    = 0.05

for epoch in range(20_000):
    total_loss = 0
    random.shuffle(training_data)

    for *features, target in training_data:
        z      = sum(features[j] * lr_weights[j] for j in range(n_inputs)) + lr_bias
        output = sigmoid(z)

        error       = target - output
        total_loss += error ** 2

        grad = error * output * (1 - output)
        for j in range(n_inputs):
            lr_weights[j] += lr_rate * grad * features[j]
        lr_bias += lr_rate * grad

    if epoch % 5000 == 0:
        print(f"  LR Epoch {epoch:>6} | Loss: {total_loss / len(training_data):.6f}")

print(f"\n  Top weighted positions (1 = most recent pick):")
indexed = sorted(enumerate(lr_weights), key=lambda x: abs(x[1]), reverse=True)
for rank, (pos, w) in enumerate(indexed[:5], 1):
    label = f"pick -{WINDOW - pos}"
    print(f"    #{rank}  {label:>8}  weight: {w:+.4f}")


# =============================================
# PHASE 2 — NEURAL NETWORK TRAINING
# Seeded from logistic regression weights.
# Uses linear output for regression prediction.
# =============================================

print("\n" + "=" * 55)
print("Phase 2: Neural Network Training")
print("=" * 55)

# Seed hidden layer 1 from LR weights + small noise
hidden_layer_1 = [
    [lr_weights[j] + random.gauss(0, 0.05) for j in range(n_inputs)]
    for _ in range(hidden_neurons)
]
hidden_layer_2 = [[he_init(hidden_neurons)  for _ in range(hidden_neurons)] for _ in range(hidden_neurons_2)]
output_neuron  = [he_init(hidden_neurons_2) for _ in range(hidden_neurons_2)]
bias_hidden_1  = [lr_bias + random.gauss(0, 0.01) for _ in range(hidden_neurons)]
bias_hidden_2  = [0.0] * hidden_neurons_2
bias_output    = 0.0

best_loss    = float('inf')
best_weights = None

for epoch in range(epochs):

    random.shuffle(training_data)
    total_loss = 0

    for *features, target in training_data:

        inputs = features

        # ── Forward pass ──────────────────────────────────────────
        h1 = []
        for i in range(hidden_neurons):
            s = sum(inputs[j] * hidden_layer_1[i][j] for j in range(n_inputs)) + bias_hidden_1[i]
            h1.append(sigmoid(s))

        h2 = []
        for i in range(hidden_neurons_2):
            s = sum(h1[j] * hidden_layer_2[i][j] for j in range(hidden_neurons)) + bias_hidden_2[i]
            h2.append(sigmoid(s))

        # Linear output (no sigmoid) — better for regression
        output_sum = sum(h2[i] * output_neuron[i] for i in range(hidden_neurons_2)) + bias_output
        output     = linear(output_sum)

        # ── Loss (MSE + L2) ───────────────────────────────────────
        error      = target - output
        l2_penalty = (
            sum(w**2 for row in hidden_layer_1 for w in row) +
            sum(w**2 for row in hidden_layer_2 for w in row) +
            sum(w**2 for w in output_neuron)
        )
        total_loss += error**2 + l2_lambda * l2_penalty

        # ── Backprop: output (linear, gradient = error) ───────────
        output_gradient    = error          # derivative of linear = 1
        old_output_weights = output_neuron[:]

        for i in range(hidden_neurons_2):
            output_neuron[i] += learning_rate * (output_gradient * h2[i] - l2_lambda * output_neuron[i])
        bias_output += learning_rate * output_gradient

        # ── Backprop: hidden layer 2 ──────────────────────────────
        old_hl2   = [row[:] for row in hidden_layer_2]
        grads_h2  = []

        for i in range(hidden_neurons_2):
            hidden_error    = output_gradient * old_output_weights[i]
            hidden_gradient = hidden_error * h2[i] * (1 - h2[i])
            grads_h2.append(hidden_gradient)

            for j in range(hidden_neurons):
                hidden_layer_2[i][j] += learning_rate * (hidden_gradient * h1[j] - l2_lambda * hidden_layer_2[i][j])
            bias_hidden_2[i] += learning_rate * hidden_gradient

        # ── Backprop: hidden layer 1 ──────────────────────────────
        for i in range(hidden_neurons):
            hidden_error    = sum(grads_h2[k] * old_hl2[k][i] for k in range(hidden_neurons_2))
            hidden_gradient = hidden_error * h1[i] * (1 - h1[i])

            for j in range(n_inputs):
                hidden_layer_1[i][j] += learning_rate * (hidden_gradient * inputs[j] - l2_lambda * hidden_layer_1[i][j])
            bias_hidden_1[i] += learning_rate * hidden_gradient

    avg_loss = total_loss / len(training_data)

    if avg_loss < best_loss:
        best_loss = avg_loss
        best_weights = {
            "hidden_layer_1": [row[:] for row in hidden_layer_1],
            "hidden_layer_2": [row[:] for row in hidden_layer_2],
            "output_neuron":  output_neuron[:],
            "bias_hidden_1":  bias_hidden_1[:],
            "bias_hidden_2":  bias_hidden_2[:],
            "bias_output":    bias_output,
        }

    if epoch % 10000 == 0:
        predicted_sample = denormalize(max(0.0, min(1.0, output)))
        print(f"  Epoch {epoch:>7} | Loss: {avg_loss:.6f} | Best: {best_loss:.6f}")

print(f"\n  Training complete. Best loss: {best_loss:.6f}")


# =============================================
# RESTORE BEST WEIGHTS
# =============================================

hidden_layer_1 = best_weights["hidden_layer_1"]
hidden_layer_2 = best_weights["hidden_layer_2"]
output_neuron  = best_weights["output_neuron"]
bias_hidden_1  = best_weights["bias_hidden_1"]
bias_hidden_2  = best_weights["bias_hidden_2"]
bias_output    = best_weights["bias_output"]


# =============================================
# SAVE MODEL
# =============================================

save_dir = "Trained_Models"
os.makedirs(save_dir, exist_ok=True)

model_name = input("\nEnter model name: ").strip()
if not model_name:
    model_name = "human_pattern"

save_path = os.path.join(save_dir, model_name + ".pkl")

# If model already exists, merge new training numbers in
existing_numbers = []
if os.path.exists(save_path):
    with open(save_path, "rb") as f:
        old = pickle.load(f)
    existing_numbers = old.get("all_numbers", [])
    print(f"  Merging with existing model ({len(existing_numbers)} previous numbers).")

merged_numbers = existing_numbers + all_numbers

model_data = {
    "NUM_MIN":        NUM_MIN,
    "NUM_MAX":        NUM_MAX,
    "WINDOW":         WINDOW,
    "n_inputs":       n_inputs,
    "hidden_neurons":   hidden_neurons,
    "hidden_neurons_2": hidden_neurons_2,
    "hidden_layer_1": hidden_layer_1,
    "hidden_layer_2": hidden_layer_2,
    "output_neuron":  output_neuron,
    "bias_hidden_1":  bias_hidden_1,
    "bias_hidden_2":  bias_hidden_2,
    "bias_output":    bias_output,
    "lr_weights":     lr_weights,
    "lr_bias":        lr_bias,
    "all_numbers":    merged_numbers,
}

with open(save_path, "wb") as f:
    pickle.dump(model_data, f)

print(f"  Model saved to: {save_path}")
print(f"  Total numbers stored in model: {len(merged_numbers)}")