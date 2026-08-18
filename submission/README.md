# AI-Based Restoration of Degraded Semiconductor Images

## KLA Semiconductor Image Restoration Challenge 2026

### Team Solution: Deep Learning Based Degradation-Aware Image Restoration

---

# 1. Overview

Semiconductor inspection systems often capture images affected by:

- Noise
- Blur
- Low resolution
- Imaging artifacts
- Unknown degradation patterns

This project presents an **AI-based degradation-aware restoration framework** that reconstructs high-quality high-resolution semiconductor images from degraded low-resolution inputs.

The proposed solution uses a **Degradation-Aware Residual U-Net v2** model that learns degradation characteristics from the input image and performs adaptive restoration.

---

# 2. Proposed Solution

The final selected model:

## DA Residual U-Net v2

Key capabilities:

- Learns degradation characteristics automatically
- Performs image denoising and restoration
- Recovers fine semiconductor structures
- Enhances spatial resolution by 2×
- Generalizes across different degradation conditions

---

# 3. Input → Restoration → Output

The system receives degraded low-resolution `.npy` images and produces restored high-resolution `.npy` images.

<p align="center">
<img src="input_output_example.png" width="900">
</p>


### Input

- Format: `.npy`
- Resolution: `128 × 128`
- Single channel grayscale
- Values normalized during inference


### Output

- Format: `.npy`
- Resolution: `256 × 256`
- Single channel grayscale
- Values constrained between `[0,1]`

---

# 4. End-to-End Workflow

The complete restoration pipeline:

<p align="center">
<img src="restoration_workflow.png" width="1000">
</p>


## Pipeline Stages

### 1. Input Loading

Reads degraded semiconductor images:

```
Input Image
128 × 128
```

---

### 2. Preprocessing

Operations:

- Load `.npy` file
- Convert to floating point
- Normalize values
- Prepare tensor format


---

### 3. Degradation Estimation

A dedicated degradation estimator extracts:

- Noise characteristics
- Blur information
- Artifact patterns

and generates a degradation embedding vector.


---

### 4. DA Residual U-Net v2 Restoration

The restoration backbone contains:

- Encoder
- Residual bottleneck blocks
- Decoder
- Skip connections
- Degradation-aware modulation


The degradation embedding guides the network to adapt restoration according to image conditions.


---

### 5. Super Resolution Module

The model performs:

```
128 × 128  →  256 × 256
```

using:

- Convolution
- PixelShuffle upsampling


---

### 6. Post Processing

Final output processing:

- Remove invalid values
- Clip range to `[0,1]`
- Convert to float32


---

# 5. Model Architecture

## DA Residual U-Net v2

Configuration:

| Parameter | Value |
|---|---:|
| Base Channels | 48 |
| Scale Factor | 2 |
| Degradation Embedding Dimension | 16 |
| Input Channels | 1 |
| Output Channels | 1 |


Main components:

```
Input
 |
Head Convolution
 |
Encoder Blocks
 |
Degradation Estimator
 |
Degradation Modulation
 |
Residual Bottleneck
 |
Decoder Blocks
 |
PixelShuffle x2
 |
Output
```

---

# 6. Repository Structure

```
submission/

│
├── run.py
│
├── requirements.txt
│
├── README.md
│
├── models/
│   |
│   └── da_resunet_sr_v2/
│       |
│       └── da_resunet_sr_v2_best.pth
│
├── input_output_example.png
│
└── restoration_workflow.png

```

---

# 7. Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

The solution requires:

- Python
- PyTorch
- NumPy
- OpenCV
- Supporting scientific libraries


---

# 8. Running the Solution

The required execution format:

```bash
python run.py <input-dir> <output-dir>
```

Example:

```bash
python run.py ../data/Test_NoisyLR/NoisyLR test_output
```

---

# 9. Input Format

The input directory should contain:

```
input-dir/

000000.npy
000001.npy
000002.npy
...
```

Each file:

```
Shape:
(H,W)

Example:
(128,128)
```

---

# 10. Output Format

The generated output directory:

```
output-dir/

000000.npy
000001.npy
000002.npy
...
```

Each output satisfies:

| Requirement | Status |
|---|---|
| Same filename as input | ✓ |
| `.npy` format | ✓ |
| Grayscale image | ✓ |
| Correct target resolution | ✓ |
| Shape `(H,W)` | ✓ |
| Range `[0,1]` | ✓ |
| No NaN values | ✓ |
| No Inf values | ✓ |

---

# 11. Validation

The submission was tested using:

- 400 test images
- Automatic batch inference
- Output format verification


Validation results:

```
Files processed : 400

Output resolution:
256 × 256

Invalid files:
0

NaN values:
0

Inf values:
0
```

---

# 12. Inference Performance

Test environment:

- NVIDIA GPU workstation
- PyTorch inference
- Batch processing pipeline


Performance:

```
Images processed : 400

Total inference time:
~23 seconds
```

Average:

```
~0.058 seconds/image
```

---

# 13. Key Advantages

✓ Degradation-aware restoration

✓ Preserves semiconductor structural details

✓ Handles unknown noise and blur patterns

✓ Improves image resolution by 2×

✓ Fully offline inference

✓ No external API dependency

✓ GPU accelerated execution

✓ Ready for automated inspection pipelines


---

# 14. Final Submission Compliance

The solution satisfies all required submission checks:

✓ run.py entry point provided

✓ Reads `.npy` input files

✓ Creates output directory automatically

✓ Generates one output per input

✓ Maintains filenames

✓ Produces valid grayscale arrays

✓ Includes model weights

✓ Includes dependency file

✓ Runs without internet access

✓ Supports NVIDIA GPU execution


---

# End-to-End AI Restoration Pipeline

**Degraded Semiconductor Image → Degradation Understanding → Adaptive Restoration → Super Resolution → Restored High Quality Image**
