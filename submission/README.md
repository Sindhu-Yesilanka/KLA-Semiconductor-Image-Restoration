# AI Based Restoration of Degraded Images

## Model
Degradation Aware Residual U-Net Super Resolution (DA Residual U-Net v2)

## Folder Structure

submission/
│
├── run.py
├── requirements.txt
├── README.md
└── models/
    └── da_resunet_sr_v2_best.pth


## Installation

Install dependencies:

pip install -r requirements.txt


## Execution

Run:

python run.py <input_dir> <output_dir>


Example:

python run.py data/Test_NoisyLR/NoisyLR output


## Input

Input files must be grayscale numpy arrays:

Shape:
(H,W)

Range:
any numeric range


## Output

Generated restored images:

- Same filename as input
- Format: .npy
- Shape: (H,W)
- Values clipped to [0,1]


## Hardware

Supports NVIDIA GPU acceleration with CUDA.
