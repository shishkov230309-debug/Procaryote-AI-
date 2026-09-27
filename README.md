# Procaryote-AI-

## Reproducible environment

Install the pinned dependencies from `requirements.txt` with Python 3.13.2. The pins were verified against the current Windows environment, including PyTorch 2.7.1 with CUDA 11.8 support.

The checked-in `checkpoints/vit_best.pth` is incompatible with this project implementation. Its keys use `vit.layers.*`, `q_proj`, `k_proj`, and `v_proj`, while the pinned Hugging Face `ViTForImageClassification` uses `vit.encoder.layer.*`, `attention.query`, `attention.key`, and `attention.value`. The checkpoint has no embedded library or version metadata, so its exact producer cannot be determined from the file alone. It must not be evaluated; retrain with the pinned environment to create a compatible checkpoint.