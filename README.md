# Pneumonia Detection using Chest X-ray Images and Machine Learning

## **Project Overview**
This project aims to classify **pneumonia** in **chest X-ray images** using **deep-learning feature extraction** with **ResNet50** and various machine learning models. The dataset used in this project is sourced from Kaggle and contains **5,863 X-ray images**, categorized into two classes:
- **Pneumonia**
- **Normal (Healthy Lungs)**

## **Dataset**
The dataset is structured into three subsets:
- **Training Set** (`train/`)
- **Testing Set** (`test/`)
- **Validation Set** (`val/`)

Each subset contains separate folders for **Pneumonia** and **Normal** cases.

### **Dataset Origin**
These **chest X-ray images** were collected from **children aged 1 to 5 years old** at **Guangzhou Women and Children’s Medical Center**. The X-ray images underwent **quality checks**, ensuring that only **clear and reliable images** were included. Diagnoses were validated by **two expert radiologists**, with a **third expert reviewing the validation set** to ensure accuracy.

### **Pneumonia Detection in X-rays**
- **Normal lungs** appear clear, without abnormal opacities.
- **Bacterial pneumonia** typically shows **localized dense white areas (lobar consolidation)** in one lung.
- **Viral pneumonia** presents **widespread interstitial patterns**, affecting **both lungs** with diffuse, less dense opacities.

## **Methodology**
Since **PySpark does not natively support CNNs or image processing**, we use **TensorFlow’s ResNet50 model** to extract **deep-learning-based feature vectors** from the X-ray images. These extracted features are then stored in a **PySpark DataFrame** and used for further classification.

### **Feature Extraction Process**
1. **Load and preprocess images** – Convert X-rays into numerical format.
2. **Extract deep-learning features using ResNet50** – Convert raw images into 2048-dimensional feature vectors.
3. **Store extracted features in PySpark** – Create a structured dataset for machine learning models.

## **Machine Learning Models Used**
After feature extraction, we train multiple machine learning classifiers:
- **Logistic Regression**: A baseline model to assess linear separability.
- **Random Forest**: A tree-based model that handles non-linear relationships well.
- **Multi-Layer Perceptron (MLP)**: A neural network classifier to capture complex patterns in the data.

## **Evaluation Metrics**
The models are evaluated using **Multiclass Classification Accuracy**, calculated as:
\[
\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}
\]
To improve performance, we also implement **hyperparameter tuning with Cross-Validation**.

## **Setup and Requirements**
### **1. Install Dependencies**
This project requires **PySpark**, **TensorFlow**, **NumPy**, and other essential libraries. Install them using:
```bash
pip install pyspark tensorflow numpy pandas -python matplotlib
```

### **2. Clone the Repository**
```bash
git clone https://github.com/gemayanna/pneumonia-xray-detection
cd pneumonia-detection
```

### **3. Download and Prepare the Dataset**
```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d dataset
```

## **Usage**
### **Run the PySpark script**
```bash
python pneumonia_detection.py
```
The script will:
- Load and preprocess X-ray images.
- Extract features using ResNet50.
- Train machine learning models.
- Evaluate model accuracy.

## **Results**
The best-performing model's accuracy will be displayed after training. The results can be further analyzed using visualization techniques.

## **License**
This project is licensed under the MIT License.

## **Authors**
- Anna Camprubí Buxeda
- Gema Novo Rego



