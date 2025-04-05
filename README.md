## 🧾 Project Description

Alzheimer’s disease (AD) is a progressive neurodegenerative disorder that affects millions of individuals worldwide. Early and accurate diagnosis is crucial for managing the disease and improving patient outcomes.

This project investigates the use of deep learning models to classify brain MRI scans for Alzheimer's detection. Convolutional Neural Networks (CNNs) are used to automate the diagnosis process by learning from imaging data.

A structured engineering workflow was implemented to ensure reliable and reproducible results. The project explores multiple CNN architectures, evaluates the effects of data augmentation, and addresses important issues such as data leakage and class imbalance. Both multi-class and binary classification strategies are examined to assess their effectiveness in real-world medical scenarios.

---

## 🎯 Objective

The main goals of this project are:

- **Training and Evaluating CNN Architectures:** Three deep learning models — ResNet50, VGG16, and EfficientNetB4 — were trained on MRI data using strict patient-level splitting to prevent data leakage.
- **Analyzing the Impact of Data Augmentation:** Models were trained on both raw and augmented data to measure the effect of preprocessing on classification accuracy.
- **Demonstrating the Risks of Data Leakage:** An experiment was conducted where patient-level separation was not enforced, highlighting how leakage can artificially boost model performance.
- **Comparing Multi-Class vs Binary Classification:** Initially, a four-class setup was used (Non-Demented, Very Mild Dementia, Mild Dementia, Moderate Dementia). A secondary experiment converted this into a binary classification task (Alzheimer vs. No Alzheimer) to compare generalization.

This project aims to provide insights into building robust, accurate, and clinically meaningful deep learning models for neurodegenerative disease detection using MRI data.

## 📁 Dataset

- **Type:** Brain MRI images labeled as either deifferent stages of Alzheimer or Non-Alzheimer
- **Source:** Public dataset available through [Kaggle](https://www.kaggle.com/datasets/ninadaithal/imagesoasis)
- **Preparation:**
  - Images resized and normalized
  - Applied augmentation: horizontal flip, rotation, zoom
  - Split into training and validation sets 

---

## 🧠 Model Architecture

- Used **Transfer Learning** with a pre-trained model (ResNet50, VGG16, EfficientNet B4)
- Replaced top classification layers with custom fully connected layers
- Regularization techniques:
  - Dropout layers to prevent overfitting
  - Early stoppping
  - Reduce learning rate when a metric has stopped improving (ReduceLROnPlateau)
- Optimizer: Adam
- Loss Function: Binary Crossentropy and CrossEntropyLoss

---

## 🛠️ Tools and Libraries

- Python
- PyTorch
- OpenCV
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

---

## 📊 Results

### 🔬 Baseline Results – Impact of Data Augmentation

To evaluate the effect of data augmentation, all three CNN models were trained both **with** and **without** augmentation using a consistent patient-level split. The goal was to assess whether standard augmentation techniques (e.g. flipping, rotation) improve classification performance.

#### 📊 Summary of Key Metrics
| Metric          | ResNet50 (no aug) | ResNet50 (aug) | VGG16 (no aug) | VGG16 (aug) | EffNetB4 (no aug) | EffNetB4 (aug) |
|-----------------|------------------:|----------------:|----------------:|-------------:|-------------------:|----------------:|
| **Validation AUC**    | 0.8061             | 0.7636          | 0.8242          | **0.8284**   | 0.6248              | 0.5125          |
| **Train AUC**         | 0.8812             | 0.8147          | **0.9901**      | 0.9760       | 0.8152              | 0.7250          |
| **Validation F1**     | 0.3216             | 0.2746          | **0.4046**      | 0.3199       | 0.2338              | 0.2241          |
| **Train F1**          | 0.3950             | 0.3048          | **0.9007**      | 0.5995       | 0.3025              | 0.2498          |
| **Validation Precision** | 0.3820         | 0.2736          | **0.4366**      | 0.3327       | 0.2757              | 0.2574          |
| **Train Precision**   | 0.5039             | 0.4315          | **0.9403**      | 0.7222       | 0.4500              | 0.3844          |
| **Validation Recall** | 0.3172             | 0.2874          | **0.3899**      | 0.3446       | 0.2574              | 0.2653          |
| **Train Recall**      | 0.3734             | 0.3007          | **0.8906**      | 0.5786       | 0.2574              | 0.2653          |

### 📌 Observations

- **Data augmentation did not improve performance** for most models. In fact, it slightly worsened AUC and F1-scores for ResNet50 and EfficientNetB4.
- **VGG16** was the only model that benefited marginally from augmentation in terms of validation AUC (0.824 → 0.828), but it came with a drop in F1-score and precision.
- All models showed signs of **overfitting**, particularly VGG16 without augmentation, which achieved very high training scores but lower validation performance.
- **EfficientNetB4** consistently underperformed compared to the other architectures, suggesting it may be too complex for this dataset or under-tuned.
- Simpler architectures like **VGG16** appear to generalize better on smaller medical datasets.


### 🔍 Summary of Experiment 1 – Effect of Data Leakage 🚨 
In this experiment, the model was trained **without enforcing patient-level separation**, meaning that MRI scans from the same individuals could appear in both the training and validation sets. This setup led to a **dramatic increase in model performance** across all metrics, including accuracy, AUC, and F1-score.

At first glance, the model appeared highly effective—achieving near-perfect results. However, these results were **misleading**. The model did not learn to detect Alzheimer's disease; instead, it **memorized patient-specific features**, resulting in **data leakage**.

This experiment underscores a critical insight in medical deep learning:

> **Without strict patient-level data partitioning, model performance may be artificially inflated and not generalizable to unseen patients.**

Such leakage can lead to overly optimistic conclusions and undermine the reliability of AI models in clinical applications.

✅ **Conclusion:** Patient-level splitting is not optional — it is essential for building trustworthy, real-world-ready diagnostic models in medical AI.


### 🔍 Summary of Experiment 2 – Binary Classification with Proper Data Splitting

In this experiment, I evaluated the model's performance on a **binary classification task** (Alzheimer vs. No Alzheimer), this time ensuring **strict patient-level data splitting** to prevent data leakage. Unlike the previous setup, images from the same patient were not present in both training and validation sets.

The model achieved a **validation AUC of 0.87** and an **Average Precision of 0.68**, indicating reasonable performance. However, the confusion matrix showed a notable number of **false negatives**, highlighting the difficulty of correctly identifying Alzheimer cases.

#### 🔑 Key Insights

- **Proper data partitioning** remains essential for producing realistic, generalizable results and avoiding misleading performance metrics.
- **Binary classification** simplified the task and improved performance compared to the earlier four-class setup, but **imbalanced data and overlapping features** still pose significant challenges for robust detection.

This experiment reinforces the need for cautious evaluation and thoughtful task design when developing deep learning models for medical diagnostics.

## 📌 Conclusions

This project provides important insights into the development of deep learning models for MRI-based Alzheimer’s disease detection. Through multiple controlled experiments, the impact of data preprocessing, task design, and data partitioning strategies was systematically explored.

### 🔑 Key Findings

- **Data Augmentation:**  
  Standard augmentation techniques (e.g., flipping, rotation) had minimal or even negative impact on validation performance. This suggests that such transformations may not be well suited for MRI data, which often requires domain-specific handling.

- **Data Leakage (Improper Patient Splitting):**  
  When patient-level separation was not enforced, the model achieved unrealistically high results (AUC = 0.9823, Accuracy = 78.73%), driven by data leakage. Proper splitting led to more realistic but lower performance (AUC = 0.8061, Accuracy = 31.72%), reinforcing the critical importance of separating patient data between training and validation sets.

- **Binary vs. Multi-Class Classification:**  
  Simplifying the classification task to two categories (Alzheimer vs. No Alzheimer) led to significantly improved performance. For example, ResNet50 achieved:
  - **Binary AUC:** 0.8714 vs **Multi-class AUC:** 0.8061  
  - **Binary Accuracy:** 80.45% vs **Multi-class Accuracy:** 31.72%  
  This shows that binary classification may generalize better in clinical contexts, although it sacrifices granularity.

### 🎓 Lessons Learned & Future Directions

- **Patient-level splitting is essential** for valid model evaluation and clinical applicability.
- **Binary classification** offers better performance but may limit diagnostic detail — hybrid approaches could be explored.
- **Augmentation strategies should be adapted** to the characteristics of medical images; alternatives include:
  - Domain-specific augmentations
  - Synthetic data generation (e.g., GANs)
- **Future improvements** could include:
  - Testing more advanced architectures (e.g., transformers for vision)
  - Using explainability tools (e.g., Grad-CAM) to understand model decisions
  - Validating on external datasets to test generalizability

This project highlights both the potential and the challenges of building reliable, AI-driven diagnostic tools in healthcare — and emphasizes the importance of rigorous methodology in medical deep learning research.

### 📋 Summary of Experimental Results
### 📋 Summary of Experimental Results

| Metric             | ResNet50 (no aug) | VGG16 (no aug) | EffNetB4 (no aug) | ResNet50 (aug) | VGG16 (aug) | EffNetB4 (aug) | ResNet50 (no patient split) | ResNet50 (binary) |
|--------------------|------------------:|---------------:|------------------:|---------------:|-------------:|----------------:|-----------------------------:|-------------------:|
| **Epochs**         | 7.50              | 4.00           | 14.50             | 8.00           | 2.50         | 6.50            | 20.00                         | 8.00               |
| **Train Loss**     | 0.4509            | 0.0629         | 0.5360            | 0.5230         | 0.2553       | 0.5990          | 0.1580                        | 0.3418             |
| **Val Loss**       | 0.6098            | 1.6967         | 0.8170            | 0.6828         | 1.1400       | 0.7674          | 0.1908                        | 0.3631             |
| **Train Accuracy** | 0.3734            | 0.8906         | 0.2968            | 0.3007         | 0.5786       | 0.2653          | 0.8441                        | 0.8362             |
| **Val Accuracy**   | 0.3172            | 0.3899         | 0.2577            | 0.2874         | 0.3446       | 0.2534          | 0.7874                        | 0.8405             |
| **Train Precision**| 0.5039            | 0.9403         | 0.4500            | 0.4315         | 0.7222       | 0.3844          | 0.9034                        | 0.6832             |
| **Val Precision**  | 0.3820            | 0.4366         | 0.2757            | 0.2736         | 0.3327       | 0.2574          | 0.8665                        | 0.7434             |
| **Train Recall**   | 0.3734            | 0.8906         | 0.2968            | 0.3007         | 0.5786       | 0.2653          | 0.8441                        | 0.4681             |
| **Val Recall**     | 0.3172            | 0.3899         | 0.2577            | 0.2874         | 0.3446       | 0.2534          | 0.7874                        | 0.4631             |
| **Train F1**       | 0.3950            | 0.9007         | 0.3025            | 0.3048         | 0.5995       | 0.2498          | 0.8632                        | 0.5371             |
| **Val F1**         | 0.3216            | 0.4046         | 0.2338            | 0.2746         | 0.3199       | 0.2241          | 0.8184                        | 0.5551             |
| **Train AUC**      | 0.8811            | 0.9901         | 0.8153            | 0.8147         | 0.9576       | 0.7250          | 0.9824                        | 0.8825             |
| **Val AUC**        | 0.8061            | 0.8242         | 0.6248            | 0.7636         | 0.8284       | 0.5125          | 0.9823                        | 0.8714             |
| **Best Epoch**     | 10                | 3              | 24                | 12             | 0            | 13              | 37                            | 11                 |



## 🙋‍♂️ Author

**Jan Dyndor**  
ML Engineer & Pharmacist   
📧 dyndorjan@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/jan-dyndor/)  
📊 [Kaggle](https://www.kaggle.com/jandyndor)

---

## 🧠 Keywords

machine learning, deep learning, healthcare, medical imaging, MRI, Alzheimer, convolutional neural network, transfer learning, classification, Resnet50, VGG16, EfficientNet B4



