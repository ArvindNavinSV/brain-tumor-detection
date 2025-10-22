# 🧠 Brain Tumor Detection using U-Net

This project applies a **2D U-Net** architecture to perform **semantic segmentation** of brain tumors from MRI scans.  
It uses the **BraTS 2020 dataset** and is implemented in **PyTorch** with GPU acceleration (tested on an NVIDIA RTX 4060 Laptop GPU).

---

## 📘 Project Overview

Brain tumor detection is a critical task in medical image analysis.  
Traditional diagnostic approaches rely on expert radiologists, but deep learning enables automated, fast, and reproducible segmentation.

In this project:

- MRI scans are pre-processed and converted from 3-D `.nii` volumes to 2-D slices.  
- A **U-Net** model is trained to predict tumor regions pixel-by-pixel.  
- The trained network can segment unseen MRI slices and highlight tumor regions automatically.

---

## 📂 Folder Structure

brain_tumor_segmentation/
│
├── data/
│ ├── BraTS2020_TrainingData/ # (Not included – download from Kaggle)
│ └── processed/ # Preprocessed 2-D images (not uploaded)
│
├── notebooks/
│ └── Brain_Tumor_Detection.ipynb # Main training & inference notebook
│
├── models/
│ └── unet_brain_tumor.pth # Saved model weights (optional)
│
├── outputs/
│ ├── example_image.png
│ ├── example_mask.png
│ └── example_prediction.png
│
├── requirements.txt
├── Deep_Machine_Handling_Report.pdf
└── README.md

yaml
Copy code

---

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
Main packages:

torch, torchvision

nibabel

opencv-python

matplotlib

numpy

tqdm

scikit-learn

🧩 Dataset
Dataset: BraTS 2020 – Brain Tumor Segmentation Challenge

Each patient folder contains 3-D MRI volumes with several modalities:

File suffix	Description
_flair.nii	Fluid-attenuated inversion recovery
_t1.nii	T1-weighted MRI
_t1ce.nii	T1 with contrast enhancement
_t2.nii	T2-weighted MRI
_seg.nii	Ground-truth segmentation mask

Segmentation label meanings:

Value	Region
0	Background
1	Necrotic / Non-enhancing tumor core
2	Edema
4	Enhancing tumor

⚙️ Pre-Processing
MRI .nii volumes are loaded with NiBabel and converted to 2-D slices.
Empty slices (no tumor) are discarded.
Each slice is resized to 128 × 128 px and normalized before training.

🧠 Model Architecture – U-Net
A classical encoder-decoder CNN with skip connections.

Encoder: convolution → ReLU → batch norm → max pool

Bridge: deepest representation

Decoder: transposed convs + concatenated skip connections

Output: 1-channel sigmoid activation producing the tumor mask

🚀 Training Setup
Parameter	Value
Optimizer	Adam
Learning rate	1 × 10⁻⁴
Loss	Binary Cross Entropy / Dice Loss
Batch size	8
Epochs	10 – 20
Hardware	NVIDIA RTX 4060 Laptop GPU

📊 Evaluation Metrics
Dice Coefficient

python
Copy code
def dice_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
IoU (Intersection over Union)
for pixel-wise overlap accuracy.

🖼️ Example Results
MRI Slice	Ground Truth Mask	Predicted Mask

🧾 How to Run
Clone the repository

bash
Copy code
git clone https://github.com/ArvindNavinSV/brain-tumor-detection.git
cd brain-tumor-detection
Install dependencies

bash
Copy code
pip install -r requirements.txt
Download dataset

From Kaggle – BraTS 2020

Place extracted folder at:

kotlin
Copy code
data/BraTS2020_TrainingData/
Run the notebook

bash
Copy code
jupyter notebook notebooks/Brain_Tumor_Detection.ipynb
🧪 Future Improvements
Extend to 3-D U-Net for full volumetric segmentation

Add attention mechanisms for better localization

Incorporate data augmentation and transfer learning

📚 References
BraTS Challenge: Menze et al., The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS), IEEE TMI 2015

U-Net Paper: Ronneberger et al., U-Net: Convolutional Networks for Biomedical Image Segmentation, MICCAI 2015

👨‍💻 Author
Arvind Navin Sekar Vasanthi
MSc Complex Adaptive Systems
Chalmers University of Technology
GitHub – ArvindNavinSV

yaml
Copy code

---

✅ **After saving and pushing:**
Your GitHub repo homepage will display this nicely formatted READM
