================================================================
 Assignment 5 (Bonus) – Image Generation Using Diffusion Models
 Roll Number : MSDS25055
 Name        : Subhan
 Course      : Deep Learning – Spring 2025
================================================================

DIRECTORY STRUCTURE
-------------------
Subhan_MSDS25055_05/
├── MSDS25055_05.py            ← Main training script
├── MSDS25055_05_test.py       ← Inference / test script
├── MSDS25055_05_allCode.py    ← All code in a single file (required)
├── test_single_sample.ipynb   ← Evaluation notebook
├── Report.pdf                 ← Written report (results & findings)
├── Readme.txt                 ← This file
└── saved_models/              ← Checkpoints saved here during training


REQUIREMENTS
------------
Python  ≥ 3.8
PyTorch ≥ 2.0   (with CUDA recommended)

Install dependencies:
    pip install torch torchvision matplotlib Pillow


DATASET LAYOUT
--------------
The dataset must be organised as:

    <dataset_root>/
        class_1/
            img001.jpg
            img002.jpg
            ...
        class_2/
            img001.jpg
            ...

Use any 5 animal class sub-folders from the provided dataset.
The script picks up to 20 images per class automatically.


TRAINING
--------
Command:
    python MSDS25055_05.py --dataset_path <path_to_dataset_root>

Key arguments:
    --dataset_path      Path to dataset root (required)
    --save_dir          Where to save checkpoints (default: saved_models)
    --epochs            Training epochs (default: 50)
    --batch_size        Batch size (default: 16)
    --lr                Learning rate (default: 2e-4)
    --T                 Diffusion time steps (default: 1000)
    --image_size        Image resolution (default: 64)
    --base_ch           U-Net base channels (default: 64)
    --num_classes       Number of animal classes (default: 5)
    --images_per_class  Images per class (default: 20)
    --save_every        Save interval in epochs (default: 10)

Example:
    python MSDS25055_05.py \
        --dataset_path /data/animals \
        --epochs 100 \
        --batch_size 32 \
        --save_dir saved_models


INFERENCE
---------
Command:
    python MSDS25055_05_test.py --model_path saved_models/ddpm_final.pth

Key arguments:
    --model_path    Path to saved .pth checkpoint (required)
    --output_path   Where to save the generated image grid (default: generated.png)
    --n_samples     Number of images to generate (default: 8)

Example:
    python MSDS25055_05_test.py \
        --model_path saved_models/ddpm_final.pth \
        --output_path results/generated.png \
        --n_samples 16


EVALUATION NOTEBOOK
-------------------
Open and run all cells in:
    test_single_sample.ipynb

Set MODEL_PATH and DATASET_PATH variables at the top of the notebook
before running.  The notebook will:
  1. Load the saved model
  2. Generate a grid of 8 new images
  3. Show the full denoising trajectory (noise → image)
  4. Show the forward noising process for one real image


OUTPUTS GENERATED DURING TRAINING
----------------------------------
saved_models/noise_progression.png   – Forward diffusion visualisation
saved_models/samples_epoch_XXXX.png  – Generated images every save_every epochs
saved_models/loss_curve.png          – Training loss graph
saved_models/ddpm_epoch_XXXX.pth     – Periodic checkpoints
saved_models/ddpm_final.pth          – Final model weights


NOTES
-----
• Loss function: Custom Hybrid L2+L1  (lambda_l2=0.8)
• Model: U-Net with sinusoidal time embedding, ResBlocks, self-attention
• Forward diffusion uses the closed-form reparameterisation (NOT direct noise addition)
• Noise schedule: Linear beta schedule (β_start=1e-4, β_end=0.02)

================================================================
