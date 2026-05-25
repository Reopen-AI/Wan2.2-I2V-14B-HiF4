# Wan2.2-I2V-A14B HiFloat4 PTQ

This repository contains the final optimized artifacts for the ICME 2026
Low-Bit-width Large-Model Quantization Challenge, Track 1.

## Base Model

- Base model: Wan2.2-I2V-A14B BF16
- Task: image-to-video generation
- Quantization: HiFloat4 W4A4 quant-dequant simulation

## Quantization Policy

- `low_noise_model`: quantized
- `high_noise_model`: quantized
- First and last Transformer blocks: BF16
- Other `nn.Linear` layers: HiFloat4 W4A4 QDQ
- Activations: HiFloat4 QDQ enabled
- Exported weights: BF16 tensors after HiFloat4 QDQ, with SmoothQuant and clip
  metadata

## Files

```text
i2v_hif4_quant_state.pt
i2v_hif4_quantized_weights_low_noise_model.pt
i2v_hif4_quantized_weights_high_noise_model.pt
```

`i2v_hif4_quant_state.pt` stores calibration metadata. The two exported-weight
files store the low-noise and high-noise expert weights separately.

## Usage

Clone the code repository and place or download these files under `outputs/`.
Then run:

```bash
LOW_QUANT_WEIGHTS=outputs/i2v_hif4_quantized_weights_low_noise_model.pt \
HIGH_QUANT_WEIGHTS=outputs/i2v_hif4_quantized_weights_high_noise_model.pt \
OUT_DIR=/path/to/generated_videos \
bash opens2v_generate.sh
```

For full reproduction instructions, see the code repository README.
