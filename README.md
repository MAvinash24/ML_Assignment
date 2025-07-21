# ML_Assignment

## Image Feature Extraction and Classification
**Machine Learning in Cyber Security (20CYS215)**  

---

## Overview  
This project explores various **image feature extraction techniques** and analyzes their impact on **classification performance** using different machine learning models. We evaluate both **traditional feature extraction methods** (HOG and LBP) and **deep learning-based feature extraction** (ResNet50) to compare their effectiveness on the **CIFAR-10 dataset**.

---

## Literature Review  
### 1. Significance of Feature Extraction in Computer Vision  
Feature extraction is essential in computer vision as it helps reduce dimensionality while preserving relevant information. Key benefits include:  
- **Dimensionality Reduction**: Converts high-resolution images into compact feature representations.  
- **Improved Classification Performance**: Enhances the ability of machine learning models to differentiate between classes.  
- **Robustness & Invariance**: Many techniques remain stable under transformations like scaling, rotation, and illumination changes.  

### 2. Conventional Feature Extraction Methods  
- **Histogram of Oriented Gradients (HOG)**: Used for object detection by computing gradient histograms in localized regions.  
- **Local Binary Patterns (LBP)**: Captures local texture variations by comparing pixel intensities.  

### 3. Deep Learning-Based Feature Extraction  
- **ResNet50**: A CNN-based feature extractor that learns hierarchical representations of images. It provides improved accuracy but requires more computational resources.  

### 4. Literature Review Comparison  
Studies show that deep learning models like **ResNet50** outperform traditional methods in classification accuracy. However, **hybrid approaches** combining handcrafted and deep features can further improve performance. Using the **entire dataset** rather than a subset would also help ResNet50 generalize better.  

---

## Implementation Details  
### 1. Dataset  
- We use the **CIFAR-10 dataset**, which contains **60,000 images across 10 classes**.  
- For faster training, we use a **reduced dataset** (1,500 training images, 500 test images).  

### 2. Feature Extraction Methods  
- **HOG** (Traditional): Extracts gradient-based edge information.  
- **LBP** (Traditional): Captures local texture patterns.  
- **ResNet50** (Deep Learning): Extracts high-level features from a pre-trained CNN model.  

### 3. Classifiers Used  
- **Logistic Regression**  
- **K-Nearest Neighbors (KNN)**  
- **Random Forest**  

### 4. Performance Evaluation  
We compare classification performance based on **accuracy** and **training time**.  

| Feature Extraction | Classifier           | Accuracy | Training Time (s) |
|-------------------|--------------------|----------|-----------------|
| HOG               | Logistic Regression | 0.348    | 3.57            |
| HOG               | KNN                 | 0.336    | 0.003           |
| HOG               | Random Forest       | 0.358    | 8.37            |
| LBP               | Logistic Regression | 0.250    | 0.05            |
| LBP               | KNN                 | 0.200    | 0.009           |
| LBP               | Random Forest       | 0.220    | 1.59            |
| ResNet50          | Logistic Regression | 0.698    | 11.45           |
| ResNet50          | KNN                 | 0.548    | 0.012           |
| ResNet50          | Random Forest       | 0.642    | 12.71           |

### 5. Key Findings  
- **ResNet50 outperforms traditional feature extraction methods**, achieving the highest accuracy.  
- **HOG performs better than LBP**, as LBP is more sensitive to the image noise.  
- **Using the full dataset** could further improve ResNet50’s accuracy and generalization.  

---

## Usage Instructions  
### Clone the repository  
```bash
git clone https://github.com/MAvinash24/ML_Assignment.git
```
```bash
cd ML_Assignment
```

### Install dependencies  
```bash
pip install -r requirements.txt
```

### Run the feature extraction and classification script  
```bash
python image_feature_extraction.py
```

### View the classification results and visualizations.  

---

## Future Improvements  
- **Train on the full CIFAR-10 dataset** to improve accuracy.  
- **Combine HOG, LBP, and ResNet50 features** for a hybrid approach.  
- **Experiment with additional deep learning models** like **VGG16 or MobileNet**.  

---

## Contributors  
- **M. Avinash**  
- **P. Deepak Sai Vighnesh**  
