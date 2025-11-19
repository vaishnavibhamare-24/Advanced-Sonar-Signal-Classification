## 🚢 Advanced Sonar Signal Classification System

Underwater Mine vs. Rock Detection using ML, PCA, XGBoost & SHAP

Detecting underwater objects from sonar signals is a critical challenge in marine safety, defense, and autonomous navigation.
This project builds a high-performance sonar classification system using machine learning, dimensionality reduction, and explainability techniques, fully deployed with a Streamlit app for real-time predictions.

### 📌 Project Highlights

✔️ Performed EDA on 2,080 sonar signal instances

✔️ Applied PCA for dimensionality reduction & noise filtering

✔️ Trained & evaluated 5+ ML models

✔️ XGBoost achieved ~95% accuracy on test data

✔️ Added SHAP explainability to interpret model predictions

✔️ Deployed with Streamlit for real-time inference (<300ms latency)

### 🧠 Tech Stack
Category	Tools
Language	Python
ML Models	XGBoost, SVM, KNN, Random Forest, Logistic Regression
Dimensionality Reduction	PCA
Explainability	SHAP
Libraries	NumPy, Pandas, Scikit-learn, XGBoost, Matplotlib, Seaborn
Deployment	Streamlit

📂 Project Structure
📦 sonar-classification
 ┣ 📁 data/
 ┃ ┗ sonar_data.csv
 ┣ 📁 notebooks/
 ┃ ┗ sonar_eda_modeling.ipynb
 ┣ 📁 models/
 ┃ ┗ xgboost_sonar_model.pkl
 ┣ 📁 app/
 ┃ ┗ streamlit_app.py
 ┣ 📁 shap_analysis/
 ┃ ┗ shap_summary.png
 ┣ README.md
 ┣ requirements.txt
 ┗ LICENSE

### 🔍 Exploratory Data Analysis (EDA)

The dataset contains 60 sonar frequency-return features and a binary target:

R → Rock

M → Mine

EDA included:

Distribution analysis of all features

Correlation heatmap

PCA variance explained

Class balance

Outlier detection


### 📉 PCA Dimensionality Reduction

Reduced 60 → 15 principal components

Retained ≈95% variance

Improved model performance and reduced overfitting

Example:

from sklearn.decomposition import PCA
pca = PCA(n_components=15)
X_pca = pca.fit_transform(X)

###  Model Training & Comparison

Multiple ML models were trained:

Model	Accuracy
Logistic Regression	83%
KNN	87%
SVM (RBF)	90%
Random Forest	92%
XGBoost	≈95% (Best)

### 🏆 Final Model: XGBoost

Optimized hyperparameters included:

max_depth

learning_rate

n_estimators

gamma

subsample

Saved model:
models/xgboost_sonar_model.pkl

### 🧩 SHAP Explainability

SHAP was used to interpret:

Global feature importance

Local explanations for individual predictions

Effects of PCA components


### 🖥️ Streamlit App (Real-Time Classifier)

The application supports:

Manual input or file upload

Instant prediction (Mine/Rock)

Prediction probabilities

SHAP-based explanation

Run the app:

streamlit run app/streamlit_app.py

⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/vaishnavibhamare-24/advanced-sonar-classification.git
cd sonar-classification

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ (Optional) Open Jupyter notebook
jupyter notebook

4️⃣ Run Streamlit app
streamlit run app/streamlit_app.py


### 📈 Results

≈95% accuracy with XGBoost

Fast inference (<300ms)

Robust due to PCA + XGBoost pipeline

Clear interpretability using SHAP

### 🚀 Future Enhancements

1. CNN using spectrogram images

2. Deploy on AWS Lambda / GCP Cloud Run

3. Add anomaly detection for unknown objects

4. Build REST API for robotics navigation systems

### 🙌 Author

Vaishnavi Bhamare
Master’s in Advanced Data Analytics, University of North Texas
AI/ML & Data Science Enthusiast
