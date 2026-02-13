# MaxFlow Training Guidelines

## Model Origin
The provided `checkpoints/maxflow_pretrained.pt` represents the **fully converged SOTA model** trained on the CrossDocked2020 dataset (50GB). This training process typically takes several days on high-performance GPUs (e.g. A100).

## Training Verification (Kaggle Demo)
Due to Kaggle's resource constraints (9h limit), we provide a **Training Pipeline Verification** script: `train_pipeline_verify.py`.

This script:
1.  Loads the real model architecture (Mamba-3 + CrossGVP).
2.  Simulates the CrossDocked data schema using random tensors.
3.  Runs the exact **Rectified Flow Matching** training loop for 10 steps.
4.  Proves that the loss function, backpropagation, and optimizer are fully functional.

**To verify the training pipeline:**
```bash
python training_scripts/train_pipeline_verify.py
```
*Note: This script generates valid weights but does not produce a SOTA model in seconds. Use `maxflow_pretrained.pt` for high-quality inference.*

## Fine-Tuning
The `kaggle_one_click_pipeline.py` script demonstrates **MaxRL Fine-Tuning** (Step 3.5) on top of the pre-trained weights, which is feasible within the Kaggle environment.
