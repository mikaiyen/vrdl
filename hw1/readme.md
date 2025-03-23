# NYCU Computer Vision 2025 Spring HW1

**StudentID**: 313554058   
**Name**: 顏紹同  

---

## Introduction

This repository contains my solution for the Homework 1 image classification task. The main goal is to classify images into 100 categories using **ResNet101** with various data augmentation techniques (MixUp, CutMix, AutoAugment, etc.) to improve generalization and achieve high accuracy on the validation/test sets.  

The code is designed to run on **Kaggle Notebook**, leveraging GPU acceleration. It includes:

- A custom dataset loader (`CustomFolderDataset`) that reads images from subfolders named 0–99.  
- ResNet101 model fine-tuned from pretrained ImageNet weights with an adapted final layer for 100 classes.  
- Advanced data augmentation methods to mitigate overfitting and enhance performance.  
- Training loop with logging of training/validation accuracy.  

---

## How to Install

1. **Kaggle Environment**  
   - Simply upload the `.ipynb` file to Kaggle, then in your notebook settings, enable **GPU** under “Accelerator”.  
   - Kaggle’s environment comes pre-installed with PyTorch, torchvision, and CUDA drivers.  
   - If you need additional libraries, you can install them via `!pip install ...` in the notebook.  

2. **Local Python Environment (Optional)**  
   - Python 3.9+  
   - PyTorch >= 1.12 (with CUDA)  
   - `torchvision`, `numpy`, `tqdm`, etc.  
   ```bash
   pip install torch torchvision tqdm numpy
   ```
   - Adjust your code or paths accordingly if you run locally instead of Kaggle.

---

## Usage

1. **Prepare Data**  
   - upload data and name it vrdlhw1data, and make sure you have the dataset structure:
     ```
     vrdlhw1data/
            ├─data/
                ├─ train/
                │   ├─ 0/
                │   ├─ 1/
                │   ...
                ├─ val/
                │   ├─ 0/
                │   ├─ 1/
                │   ...
                └─ test/
                    ├─ ...
     ```

2. **Running the Notebook**  
   - Open the Kaggle Notebook, locate the `train` or main code cell, and run it.  
   - The script will:
     1. Import dependencies  
     2. Load the dataset and define data transforms  
     3. Build ResNet101 with a custom final layer for 100 classes  
     4. Train for a specified number of epochs, printing out training/validation metrics  
     5. Save the best model weights (e.g., `best_model.pt`)

3. **Inference**  
   - Load `best_model.pt` in a separate inference cell.  
   - Run your predictions on the test folder, generating `prediction.csv`.

---

## Performance Snapshot

- **Public Leaderboard**:  
  - Achieved validation accuracy of around **89%**.  

- **Training Curve**:  
  - The model typically converged in ~15–20 epochs with advanced data augmentation such as MixUp or CutMix.  
  - Label smoothing and weight decay helped manage overfitting.

---

## File Structure

```
.
├── 313554058.ipynb     # Main Kaggle Notebook
├── README.md                    # This README
├── data/                        # Dataset folder (train/val/test)
├── best_model.pt      # Trained model weights
└── ...
```

---

## References

1. [PyTorch ResNet101 Documentation](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet101.html)  
2. [MixUp Paper](https://arxiv.org/abs/1710.09412)  
3. [CutMix Paper](https://arxiv.org/abs/1905.04899)  

---

**End of README**