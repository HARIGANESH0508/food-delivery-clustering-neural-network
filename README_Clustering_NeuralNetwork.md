# 🚚 Food Delivery Clustering & Neural Network Prediction

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Neural%20Network-red?logo=tensorflow)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> Groups food delivery orders into meaningful clusters using K-Means and Hierarchical Clustering, then predicts fast vs delayed delivery using a tuned Neural Network — all built on real-world features like distance, traffic, weather, and order time.

---

## 📌 Project Highlights

- ✅ K-Means Clustering with Elbow Method to find optimal number of clusters
- ✅ Hierarchical Clustering with Dendrogram visualisation
- ✅ 3D Scatter Plot (Delivery Time vs Distance vs Traffic) using Plotly
- ✅ Neural Network (TensorFlow/Keras) to classify Fast vs Delayed deliveries
- ✅ Hyperparameter tuning — tested different layers, neurons, and learning rates
- ✅ Feature engineering with Haversine distance and Rush Hour flag

---

## 📁 Project Structure

```
food-delivery-clustering-neural-network/
├── k_means_hierarchical_clustering.ipynb   # Main notebook
├── Food_Delivery_Time_Prediction.csv        # Dataset (200 records)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/food-delivery-clustering-neural-network.git
cd food-delivery-clustering-neural-network

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open and run the notebook
jupyter notebook k_means_hierarchical_clustering.ipynb
```

---

## 📊 Dataset

**Same dataset as Food Delivery Time Prediction project**
**Size:** 200 rows × 15 columns | No missing values

| Column | Description |
|--------|-------------|
| `Distance` | Delivery distance (standardized) |
| `Delivery_Time` | Actual delivery time (target) |
| `Traffic_Conditions` | Encoded: 0=Low, 1=Medium, 2=High |
| `Weather_Conditions` | Encoded: 0=Clear, 1=Fog, 2=Rain, 3=Storm |
| `Delivery_Person_Experience` | Years of experience (1–10) |
| `Restaurant_Rating` | Rating out of 5.0 |
| `Customer_Rating` | Rating out of 5.0 |
| `Order_Cost` | Order value in rupees |
| `Tip_Amount` | Tip in rupees |
| `Order_Time` | Morning / Afternoon / Evening / Night |

---

## 🔧 Feature Engineering

| Feature | How Created |
|---------|-------------|
| `Distance_km` | Haversine formula from Customer & Restaurant GPS coordinates |
| `Rush_Hour` | 1 if order placed in Morning / Evening / Night |
| `Non_Rush_Hour` | 1 if order placed in Afternoon |
| `Order_Time_*` | One-hot encoded time-of-day columns |
| `Order_Priority_encoded` | Label encoded order priority |

---

## 🧠 Full Pipeline

```
Phase 1 → Data Preprocessing & Feature Engineering
          - Label encode: Weather, Traffic, Vehicle Type
          - Haversine distance from GPS coordinates
          - Rush Hour / Non Rush Hour flags
          - One-hot encode Order_Time

Phase 2 → K-Means Clustering
          - Elbow Method: test k = 2 to 9 clusters
          - Scaled features: Delivery_Time, Distance_km,
            Traffic_Conditions, Weather_Conditions
          - Optimal clusters: k = 4
          - Scatter plots: Distance vs Delivery Time
                           Traffic vs Delivery Time
                           Weather vs Delivery Time
                           Customer vs Restaurant location map

Phase 3 → Hierarchical Clustering
          - Ward linkage method
          - Dendrogram visualisation
          - 3D Scatter Plot (Plotly): Delivery Time × Distance × Traffic

Phase 4 → Neural Network (TensorFlow/Keras)
          - Target: Fast (1) if Delivery_Time > median, else Delayed (0)
          - Architecture: Dense(20, relu) → Dense(10, relu) → Dense(1, sigmoid)
          - Optimizer: Adam | Loss: Binary Crossentropy
          - Epochs: 20 | Batch size: 32

Phase 5 → Hyperparameter Tuning
          - Tested: layers = [1, 2], neurons = [16, 32], lr = [0.001, 0.01]
          - Best model selected by highest test accuracy
```

---

## 📈 Results

### K-Means Clustering

| Setting | Value |
|---------|-------|
| Features used | Delivery_Time, Distance_km, Traffic_Conditions, Weather_Conditions |
| Optimal clusters (Elbow) | **k = 4** |
| Scaling | StandardScaler applied before clustering |

**Cluster Interpretation:**

| Cluster | Pattern |
|---------|---------|
| 0 | Short distance, low traffic — Fast deliveries |
| 1 | Long distance, high traffic — Slow deliveries |
| 2 | Medium distance, rainy weather — Moderate delays |
| 3 | Night orders, high experience riders — Varies |

---

### Neural Network Results

| Metric | Value |
|--------|-------|
| Accuracy | ~0.55 – 0.65 |
| Precision | ~0.57 |
| Recall | ~0.50 |
| F1-Score | ~0.53 |

> **Note:** The dataset has only 200 rows which limits neural network performance. Results improve with more data. The hyperparameter tuning loop tested 8 configurations to find the best combination of layers, neurons, and learning rate.

---

### Hyperparameter Tuning — Configurations Tested

| Layers | Neurons | Learning Rate |
|--------|---------|---------------|
| 1 | 16 | 0.001 |
| 1 | 16 | 0.01 |
| 1 | 32 | 0.001 |
| 1 | 32 | 0.01 |
| 2 | 16 | 0.001 |
| 2 | 16 | 0.01 |
| 2 | 32 | 0.001 |
| 2 | 32 | 0.01 |

---

## 📊 Visualisations Included

| Chart | Description |
|-------|-------------|
| Elbow Curve | Inertia vs number of clusters (k=2 to 9) |
| Scatter: Distance vs Delivery Time | Colour-coded by cluster |
| Scatter: Traffic vs Delivery Time | Colour-coded by cluster |
| Scatter: Weather vs Delivery Time | Colour-coded by cluster |
| Location Map | Customer & Restaurant GPS points by cluster |
| Dendrogram | Hierarchical clustering tree (Ward method) |
| 3D Scatter | Delivery Time × Distance × Traffic (Plotly interactive) |

---

## 🛠️ Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
haversine
scipy
plotly
tensorflow
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🔮 Future Improvements

- Increase dataset size beyond 200 rows for better neural network training
- Try **DBSCAN** clustering for non-circular cluster shapes
- Add **Dropout layers** to reduce overfitting in the neural network
- Use **GridSearchCV with KerasClassifier** for more systematic tuning
- Visualise cluster centroids on a real map using Folium

---

## 👤 Author

**Hari Ganesh**
- Course: Machine Learning
- 📧 [hariganesh4567@gmail.com]
---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.
