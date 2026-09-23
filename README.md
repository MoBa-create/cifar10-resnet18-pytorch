CIFAR-10 Image Classification with PyTorch: From Custom CNNs to ResNet-18

An end-to-end computer vision project demonstrating iterative improvement on the CIFAR-10 dataset using PyTorch. The repository progresses from a simple baseline Convolutional Neural Network (CNN) to an optimized custom architecture and fine-tuned ResNet-18 via Transfer Learning, reaching 95.45% test accuracy.

📊 Performance Benchmark

Model

Architecture Highlights

Image Resolution

Training Epochs

Test Accuracy

SimpleCNN

2-Layer Baseline CNN

32x32

10

69.34%

ImprovedCNN

Conv Blocks + BatchNorm + Dropout + CosineAnnealingLR

32x32

15

82.28%

ResNet-18

Pre-trained Transfer Learning + Image Upscaling

128x128

12

95.45%

🔑 Key Engineering Highlights

Iterative Architectural Progression:
Developed three distinct performance tiers to evaluate capacity vs. accuracy tradeoffs.

Transfer Learning Optimization:
Fine-tuned standard ResNet-18 weights pretrained on ImageNet by adapting the final fully-connected (fc) classification head to 10 classes.

Resolution Rescaling Strategy:
Upscaled CIFAR-10 images from $32 \times 32$ to $128 \times 128$, matching ImageNet feature distribution and unlocking deep spatial representations.

Regularization & Optimization:
Employed AdamW optimizer, CosineAnnealingLR scheduler, Batch Normalization, and data augmentation (RandomCrop, RandomHorizontalFlip).

Cross-Platform & Path Safety:
Enforced absolute directory resolving (os.path.abspath) and platform-independent execution for safe CUDA execution on Windows environment.

📁 Repository Structure

├── data/                         # CIFAR-10 dataset directory (ignored by git)
├── .gitignore                    # Git exclusions file
├── requirements.txt              # Project dependencies
├── README.md                     # Project documentation
├── train_cifar10.py              # Baseline SimpleCNN training script
├── train_improved_cifar10.py     # ImprovedCNN training script
├── train_resnet18.py             # Transfer Learning ResNet-18 training script
├── predict_random.py             # Inference script for ImprovedCNN
└── predict_random_resnet.py      # Inference script & visualizer for ResNet-18


💻 Environment & Hardware Specs

Framework: PyTorch & Torchvision

Language: Python 3.13

Hardware Acceleration: NVIDIA GeForce RTX 5070 Ti Laptop GPU (CUDA)

🚀 Getting Started

1. Setup & Installation

Clone the repository and install the dependencies:

git clone https://github.com/MoBa-create/cifar10-cnn-classifier.git
cd cifar10-cnn-classifier
pip install -r requirements.txt


2. Model Training

To train the state-of-the-art ResNet-18 model:

python train_resnet18.py


3. Inference & Visualization

Run model predictions on 10 random test samples with visual evaluation grid output:

python predict_random_resnet.py
