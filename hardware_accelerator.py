import pyrtl
import numpy as np
import joblib
from sklearn.datasets import load_iris

# Load quantized model
data = np.load("quantized_model.npz")
w1, b1, w2, b2, w3, b3 = data['w1'], data['b1'], data['w2'], data['b2'], data['w3'], data['b3']

# Define hardware
pyrtl.reset_working_block()

# Inputs (4 features, 8-bit fixed-point)
# We'll pack 4 x 8-bit into a 32-bit input for simplicity
features = pyrtl.Input(32, 'features')

# --- Hardened Hardware Logic (Professional Grade) ---
def mac_unit(inputs, weights, bias):
    """
    High-precision MAC unit: 
    - Inputs: 8-bit signed
    - Products: 16-bit signed
    - Accumulator: 24-bit signed (prevents intermediate overflow)
    """
    # Initialize with bias (scaled by 16)
    mac = pyrtl.Const(int(bias) << 4, 24)
    for i, (inp, weight) in enumerate(zip(inputs, weights)):
        w_const = pyrtl.Const(int(weight), 8)
        # 8x8 signed multiply -> 16-bit product
        prod = pyrtl.signed_mult(inp, w_const)
        # Sign-extend and accumulate in 24-bit
        mac = mac + prod.sign_extended(24)
    return mac

def apply_rounding_and_saturation(mac, signed=False):
    """
    Biased Rounding (Round-to-Nearest) + Saturation:
    - Adds 2^(shift-1) before shifting for unbiased truncation.
    - Saturation prevents bit-wraparound.
    """
    # Add 0.5 (8 in Q4.4) for round-to-nearest
    mac_rounded = mac + pyrtl.Const(8, 24)
    
    # MSB for ReLU / Sign check
    is_neg = mac_rounded[23]
    
    # Right shift by 4 (divide by 16)
    val_shifted = mac_rounded[4:24] 
    
    # Saturation Logic
    sign_bit = val_shifted[19]
    upper_bits = val_shifted[8:]
    
    if signed:
        # In-range check for [-128, 127]
        is_in_range = pyrtl.select(sign_bit, pyrtl.and_all_bits(upper_bits), ~pyrtl.or_all_bits(upper_bits))
        clipped_val = pyrtl.select(sign_bit, pyrtl.Const(-128, 8), pyrtl.Const(127, 8))
        return pyrtl.select(is_in_range, val_shifted[0:8], clipped_val)
    else:
        # ReLU + Positive Saturation
        is_too_large = pyrtl.or_all_bits(val_shifted[8:])
        val_8bit = pyrtl.select(is_too_large, pyrtl.Const(127, 8), val_shifted[0:8])
        return pyrtl.select(is_neg, pyrtl.Const(0, 8), val_8bit)

# --- Pipelined Sequential Architecture (4-Stage) ---

# Stage 0: Input Buffer (Synchronous)
reg_f0 = pyrtl.Register(8, 'reg_f0')
reg_f1 = pyrtl.Register(8, 'reg_f1')
reg_f2 = pyrtl.Register(8, 'reg_f2')
reg_f3 = pyrtl.Register(8, 'reg_f3')

reg_f0.next <<= features[24:32]
reg_f1.next <<= features[16:24]
reg_f2.next <<= features[8:16]
reg_f3.next <<= features[0:8]

# Stage 1: Layer 1 Computation -> Register
layer1_comb = [apply_rounding_and_saturation(mac_unit([reg_f0, reg_f1, reg_f2, reg_f3], w1[i, :], b1[i]), signed=False) for i in range(8)]
reg_l1 = [pyrtl.Register(8, f'reg_l1_{i}') for i in range(8)]
for r, c in zip(reg_l1, layer1_comb):
    r.next <<= c

# Stage 2: Layer 2 Computation -> Register
layer2_comb = [apply_rounding_and_saturation(mac_unit(reg_l1, w2[i, :], b2[i]), signed=False) for i in range(4)]
reg_l2 = [pyrtl.Register(8, f'reg_l2_{i}') for i in range(4)]
for r, c in zip(reg_l2, layer2_comb):
    r.next <<= c

# Stage 3: Layer 3 Computation -> Output Register
layer3_comb = [apply_rounding_and_saturation(mac_unit(reg_l2, w3[i, :], b3[i]), signed=True) for i in range(3)]
reg_out = [pyrtl.Register(8, f'reg_out_{i}') for i in range(3)]
for r, c in zip(reg_out, layer3_comb):
    r.next <<= c

# Split Outputs: 3x8-bit wires (Stable outputs from Stage 3)
score0_out = pyrtl.Output(8, 'score0')
score1_out = pyrtl.Output(8, 'score1')
score2_out = pyrtl.Output(8, 'score2')

score0_out <<= reg_out[0]
score1_out <<= reg_out[1]
score2_out <<= reg_out[2]

# --- Pipelined Simulation & High-Fidelity Verification ---
sim_trace = pyrtl.SimulationTrace()
sim = pyrtl.Simulation(tracer=sim_trace)

import joblib
model, scaler, X_test_ref, y_test_ref = joblib.load("software_model.pkl")

correct = 0
total = len(X_test_ref)
conf_matrix = np.zeros((3, 3), dtype=int)
max_margin_reduction = 0.0
PIPELINE_LATENCY = 4 # Cycles from input arrival to stable output

print("\n=== PIPELINED SEQUENTIAL VALIDATION (4-STAGE) ===")
print(f"Evaluating full dataset with {PIPELINE_LATENCY}-cycle latency handling...\n")

for i in range(total):
    scaled_sample = X_test_ref[i]
    q_input = np.clip(np.round(scaled_sample * 16), -128, 127).astype(np.int8)
    packed_input = int.from_bytes(q_input.tobytes(), byteorder='big', signed=False)
    
    # Step 1: Provide Input
    sim.step({features.name: packed_input})
    
    # Step 2: Flush Pipeline (N cycles)
    # We step N-1 more times to allow the specific input to reach the output registers
    for _ in range(PIPELINE_LATENCY - 1):
        sim.step({features.name: packed_input}) 
    
    # Step 3: Collect scores from discrete output wires
    scores = []
    for out_w in [score0_out, score1_out, score2_out]:
        val = sim.value[out_w]
        if val > 127: val -= 256
        scores.append(val)
    
    # Classification Logic
    hw_pred = np.argmax(scores)
    sw_probs = model.predict_proba([scaled_sample])[0] 
    sw_pred = np.argmax(sw_probs)
    
    conf_matrix[sw_pred, hw_pred] += 1
    match = (hw_pred == sw_pred)
    if match: correct += 1
    
    # Top-2 Margin Analysis (Probability domain)
    sw_sorted = np.sort(sw_probs)
    sw_margin = sw_sorted[-1] - sw_sorted[-2]
    # Softmax-like normalization for HW scores to compare margins
    exp_scores = np.exp(np.array(scores) / 16.0)
    hw_probs = exp_scores / np.sum(exp_scores)
    hw_sorted = np.sort(hw_probs)
    hw_margin = hw_sorted[-1] - hw_sorted[-2]
    
    margin_diff = sw_margin - hw_margin
    max_margin_reduction = max(max_margin_reduction, margin_diff)
    
    print(f"[Sample {i:2d}] HW={hw_pred} | SW={sw_pred} | scores={scores} | {'OK' if match else 'MISMATCH'}")
    
    if not match:
        print(f"   --- DEBUG: Margin Drift ---")
        print(f"   SW Margin: {sw_margin:.2f} | HW Margin: {hw_margin:.2f}")

print(f"\n--- Pipelined Verification Summary ---")
print(f"Match Accuracy: {correct}/{total} ({correct/total*100:.2f}%)")
print(f"Max Top-2 Margin Reduction: {max_margin_reduction:.4f}")
print(f"Pipeline Depth: {PIPELINE_LATENCY} stages")
print("\nConfusion Matrix:")
print(conf_matrix)

# --- Export Assets ---
with open("accelerator.v", "w") as f:
    pyrtl.output_to_verilog(f)
with open("trace.vcd", "w") as f:
    sim_trace.print_vcd(f)
print("\nSynthesizable Sequential RTL and VCD generated.")
