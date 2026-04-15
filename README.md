# 🏠 House Price Prediction using Deep Neural Networks

## 📌 Project Overview

This project implements a **deep learning-based regression system** to predict house prices in Indian cities. It focuses on comparing different **optimization techniques and regularization methods** to improve model performance and generalization.

The system also includes a **web-based interface (Flask)** for real-time price prediction.

---

## 🎯 Objectives

* Build a deep neural network for house price prediction
* Compare optimizers: **SGD, Adam, RMSProp**
* Apply regularization techniques: **L1, L2, Dropout**
* Use **Batch Normalization** for training stability
* Implement **learning rate scheduling**
* Evaluate model using regression metrics
* Deploy model via a **Flask web application**

---

## 🧠 Model Features

* Deep Neural Network (DNN)
* Xavier & He initialization comparison
* Optimizer comparison:

  * SGD
  * Adam
  * RMSProp
* Regularization:

  * L1 & L2
  * Dropout
* Batch Normalization
* Learning Rate Scheduling

---

## 📊 Dataset

* Indian housing dataset (multi-city)
* Features include:

  * City, Locality Tier
  * BHK, Bathrooms
  * Area (Super & Carpet)
  * Floor details
  * Amenities (Lift, Parking, Gated Society)
  * Distance metrics
  * Crime index

👉 Dataset represents **apartment/flat properties**

---

## ⚙️ Tech Stack

* Python
* TensorFlow / Keras
* Scikit-learn
* Pandas / NumPy
* Flask (Web App)
* HTML, CSS, JavaScript (Frontend)

---

## 📈 Model Performance

* R² Score: ~0.96
* MAE: ~9 Lakhs
* RMSE: ~12 Lakhs

✔ Indicates strong predictive performance

---

## 🌐 Web Application Features

* Modern UI with glassmorphism design
* Dynamic city selection
* Optional advanced inputs
* Smart defaults based on locality tier
* Animated price prediction
* Price displayed in:

  * Full INR format
  * Lakhs / Crores
* Recommended price range (based on RMSE)
* Confidence score
* Loading animation

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Train the model

Run the Jupyter Notebook:

```bash
train.ipynb
```

---

### 3. Start Flask app

```bash
python app.py
```

---

### 4. Open in browser

```
http://127.0.0.1:5000/
```

---

## 📂 Project Structure

```
House-Price-Prediction/
│
├── model/
│   ├── best_model.keras
│   ├── scaler.pkl
│   ├── y_scaler.pkl
│
├── templates/
│   └── index.html
│
├── app.py
├── house_price_model.ipynb
├── requirements.txt
└── README.md
```

---

## 🧪 Evaluation Metrics

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

---

## 📌 Key Highlights

* Combines **ML research + real-world application**
* Demonstrates **optimization & regularization effects**
* Includes **end-to-end pipeline (training → deployment)**

---

## 🔮 Future Improvements

* Add map-based location input
* Improve dataset with real-world listings
* Deploy online (Render / AWS)
* Add model explainability (SHAP)


---

## 📄 License

This project is for academic and educational purposes.
