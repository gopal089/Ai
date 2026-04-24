# PyRTL-Based Quantized Neural Network Hardware Accelerator

## Project Overview
This project demonstrates a complete digital design pipeline for a **Quantized Neural Network (QNN) Hardware Accelerator**. It bridges the gap between software-based machine learning and gate-level VLSI design, implementing a signed 8-bit fixed-point inference engine for the Iris dataset.

### Key Features
- **Hardened Datapath**: 8x8 -> 16-bit products accumulated into a 24-bit widening register to prevent precision loss.
- **Biased Rounding**: Implements **Round-to-Nearest** logic (`(acc + 0.5) >> shift`) to minimize quantization bias.
- **RTL Generation**: Synthesizable Verilog (`accelerator.v`) optimized for low-area edge inference.
- **High-Fidelity Verification**: 96.67% HW–SW agreement on 30-sample test set (Δ = 3.33% vs FP32 baseline).

## Hardware Architecture
The design utilizes a **fully combinational spatially unrolled datapath**:
- **Latency**: Fully combinational inference per simulation step; no pipelining.
- **Numerical Robustness**: Dedicated **Signed Saturation** logic clips values at +127/-128, preventing bit-wraparound errors.
- **Endianness**: Prediction bus is 24-bit Big-Endian: `[Score0(23:16), Score1(15:8), Score2(7:0)]`.

## Synthesis Metrics (Yosys)
Extracted via the **Yosys Open SYnthesis Suite**:
- **Total Standard Cells**: 384
- **Hardware Multipliers**: 74
- **Logic Footprint**: Optimized low-area design by eliminating floating-point logic and using a compact 8-bit datapath.

## Project Structure
- `train_model.py`: Training, L2 regularization, and 8-bit quantization.
- `hardware_accelerator.py`: PyRTL implementation, simulation, and Verilog export.
- `accelerator.v`: Synthesizable Verilog RTL.
- `trace.vcd`: Simulation waveforms for GTKWave/VCD analysis.

### High-Fidelity Validation
- **Max Top-2 Margin Reduction**: 0.14 (Normalized)
- **Error Profile**: Mismatches are confined to decision boundaries where margins are < 0.2; no systemic bias.

---
*Developed as a VLSI-ML Co-Design showcase.*
# Ai
