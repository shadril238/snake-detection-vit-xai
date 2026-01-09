# Snake Classification with Custom ViT + XAI
Custom Vision Transformer (ViT) with hybrid CNN+ViT feature learning, Optuna hyperparameter tuning, and explainable AI (Integrated Gradients + LIME) for snake image classification.

## Features

- **Dataset preprocessing**
  - Class distribution analysis
  - Sample visualization
  - Corrupt image detection
  - Image size statistics and outlier visualization

- **Stratified dataset partition**
  - Train/Validation/Test split (balanced across classes)

- **Feature extraction**
  - CNN embeddings (e.g., EfficientNet family using `timm`)
  - ViT embeddings (e.g., ViT-Base using `timm`)
  - **Hybrid features** by concatenating CNN + ViT embeddings

- **Modeling (3 families)**
  1. **Deep learning on extracted CNN features** (MLP classifier)
  2. **Hybrid deep learning on extracted CNN+ViT features** (MLP classifier)
  3. **End-to-end ViT fine-tuning** on images

- **Hyperparameter tuning (Optuna)**
  - Tunes the **best-performing model family** among the three

- **Custom Vision Transformer (Custom ViT)**
  - Tunable classification head
  - Selective unfreezing of transformer blocks
  - Hyperparameter tuning of training + architecture
  - Final training and evaluation with optional TTA

- **Explainable AI (XAI)**
  - **Integrated Gradients** (Captum)
  - **LIME** explanations on test images
  - Visual outputs: raw image, heatmap, overlay, and probability summaries

- **Experiment tracking & artifact saving**
  - Model checkpoints, histories, metrics, confusion matrices, and configs saved to Google Drive


