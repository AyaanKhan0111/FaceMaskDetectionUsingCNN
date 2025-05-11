# Face Mask Detection System

Developed by:
- Muhammad Qasim (22I-1994)
- Ayaan Khan (22I-2066)
- Abubakar Nadeem (22I-2003)
- Ahmed Mehmood (22I-1915)

## 📌 Overview

This project presents a **real-time face mask detection system** built using computer vision and deep learning techniques. It can determine whether individuals in an image or video stream are **wearing a mask**, **not wearing a mask**, or wearing it **incorrectly**. The system is designed for public safety monitoring during respiratory outbreaks and integrates a lightweight, efficient model suitable for real-time applications.

---

## 📁 Dataset

- **Source**: [Kaggle - Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
- **Classes**: 
  - `with_mask`
  - `without_mask`
- **Samples**: ~3725 images per class
- **Split**: 80% Training, 20% Testing (stratified)

---

## 🧪 Image Preprocessing Pipeline

The system uses several digital image processing techniques:
1. Resize images to `224x224`
2. Convert color space: `BGR → RGB` (for model), `BGR → Grayscale` (for preprocessing)
3. Histogram Equalization (contrast enhancement)
4. Gaussian Blur (noise reduction)
5. Otsu Thresholding (adaptive binarization)
6. Morphological Closing (connect components)
7. Masking (focus on facial features)
8. MobileNetV2-specific preprocessing (normalization)

---

## 🧠 Model Architecture

### 🔹 Base Model: MobileNetV2 (Transfer Learning)

Key features:
- Inverted Residual Blocks
- Depthwise Separable Convolutions
- Linear Bottlenecks
- Lightweight and optimized for mobile/edge devices

### 🔧 Fine-Tuning & Optimization
- Learning rate scheduling
- Early stopping
- Last 30% layers unfrozen for fine-tuning
- Dropout (20%) to reduce overfitting
- Data augmentation (flips, rotations, zooms, contrast)

---

## 🕵️‍♂️ Face Detection

- **Model**: SSD with ResNet-10 backbone
- **Files**:
  - `deploy.prototxt`
  - `res10_300x300_ssd_iter_140000_fp16.caffemodel`
- **Library**: OpenCV DNN module for real-time face detection

---

## 🌐 Web Application (Flask)

Features:
- Live webcam mask detection
- Upload and analyze images
- Detection stats: total faces, with/without mask, confidence scores
- Visualization: bounding boxes & labels

---

## 📊 Results

- **High accuracy** in binary classification
- **Real-time detection** on standard laptops
- Robust against varying lighting conditions

### Sample Outputs:
- ✅ Detection of individuals wearing masks (green boxes)
- ❌ Detection of individuals without masks (red boxes)
- 📷 Live detection interface through webcam

---

## ⚠️ Limitations

- Lower accuracy with:
  - Non-mask facial occlusions
  - Extreme head poses
  - High-res video streams on low-end hardware

---

## 🚀 Future Work

### Improvements:
- Multi-class classification (e.g., surgical, cloth, N95 masks)
- Integration with crowd monitoring or CCTV systems
- Deployment on edge devices (e.g., Raspberry Pi)
- Use of attention mechanisms

### Additional Features:
- Long-term compliance statistics
- Access control system integration
- Mobile app interface

---

## ✅ Conclusion

This project demonstrates an effective real-time face mask detection system using deep learning. Leveraging **MobileNetV2** and advanced **image processing techniques**, it offers an accurate and efficient solution deployable via a **Flask web interface**. Ideal for public safety and health monitoring systems.

---

## 🛠️ Tech Stack

- Python
- OpenCV
- TensorFlow / Keras
- Flask
- Numpy, Matplotlib, Scikit-learn


