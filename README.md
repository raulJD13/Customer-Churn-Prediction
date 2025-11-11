# Customer Churn Prediction

## 📋 Table of Contentsf

- [Overview](#overview)
- [Key Insights](#key-insights)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Leakage Journey](#data-leakage-journey)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project demonstrates a complete Data Science workflow, from raw data to deployed API. Starting with 9 separate CSV files simulating a complex database, it follows a rigorous process of:

- **Data integration** and cleaning
- **Feature engineering** with domain insights
- **Model development** and rigorous debugging
- **API deployment** with FastAPI

The final RandomForestClassifier achieves **92% AUC** and successfully identifies customers at high risk of leaving the platform.

## 💡 Key Insights

### The Shipping Cost Discovery

After training the final model, feature importance analysis revealed a surprising insight:

**The #1 predictor of churn is NOT purchase frequency or monetary value—it's `avg_freight_value` (shipping cost).**

#### Why This Matters

- **The Problem**: The frequency feature was useless because over 95% of customers are one-time buyers
- **The Pattern**: A complex, non-linear relationship exists between shipping cost and churn

#### The Churn Zones

| Zone | Shipping Cost Range | Churn Rate | Interpretation |
|------|---------------------|------------|----------------|
| 🚨 **Danger Zone** | $12 - $18 | **67%** | Medium-low fees perceived as unfair "penalty" |
| 🏅 **Sweet Spot** | $18 - $27 | **47%** | Medium-high fees for "locked-in" remote customers |

This non-linear pattern is exactly what tree-based models excel at finding—something a linear model would completely miss.

## 📈 Model Performance

| Model | AUC-ROC | Recall (Churn) | Precision (Churn) |
|-------|---------|----------------|-------------------|
| **Random Forest** | **0.9214** | **0.88** | **0.85** |
| Logistic Regression | 0.6057 | 0.63 | 0.66 |

**What this means**: The model correctly identifies 88% of all at-risk customers while maintaining 85% precision—meaning alerts are trustworthy and actionable for marketing teams.

## 🗂️ Project Structure

```
Customer-Churn-Prediction/
│
├── datasets/                          # Raw CSV files (not included in repo)
│   ├── olist_customers_dataset.csv
│   ├── olist_orders_dataset.csv
│   └── ... (7 more files)
│
├── notebooks/
│   ├── 01_ETL_and_Cleaning.ipynb     # Data integration & cleaning
│   ├── 02_EDA_and_Churn_Definition.ipynb  # Exploratory analysis & target definition
│   ├── 03_Feature_Engineering.ipynb  # Advanced feature creation
│   └── 04_Modeling_and_Evaluation.ipynb   # Model training & evaluation
│
├── models/
│   └── rf_churn_model.joblib         # Trained model (generated)
│
├── main.py                            # FastAPI application
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

### Notebook Breakdown

#### 01_ETL_and_Cleaning.ipynb
- Loads all 9 source CSV files from the Olist dataset
- Performs `pd.merge()` joins to create a Master Transactional DataFrame
- Cleans data and exports `df_master_transaccional.parquet`

#### 02_EDA_and_Churn_Definition.ipynb
- Conducts exploratory data analysis (EDA)
- Defines churn: *A customer who has not purchased in 180 days*
- Transforms dataset from (1-row-per-order) to (1-row-per-customer)
- Creates initial RFM (Recency, Frequency, Monetary) features

#### 03_Feature_Engineering.ipynb
Engineers advanced behavioral features:
- `tenure_days` - Customer lifetime
- `avg_payment_value` - Average order value
- `avg_items_per_order` - Purchase basket size
- `main_category` - Most-purchased product category
- Exports `model_input.parquet`

#### 04_Modeling_and_Evaluation.ipynb
- Builds scikit-learn Pipeline with ColumnTransformer
- Trains and compares Logistic Regression vs Random Forest
- Evaluates metrics (AUC, Recall, Precision, F1)
- Extracts and visualizes Feature Importance
- Validates shipping cost hypothesis
- Saves `rf_churn_model.joblib`

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

2. **Create virtual environment** (recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the dataset**
   - Download the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) from Kaggle
   - Create a `datasets/` folder in the project root
   - Extract all 9 CSV files into `datasets/`

## 📊 Usage

### Run the Analysis Pipeline

Execute notebooks in order:

```bash
jupyter notebook
# Open and run: 01 → 02 → 03 → 04
```

Notebook 04 will generate `models/rf_churn_model.joblib`

### Launch the API

```bash
uvicorn main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

### Example API Request

```python
import requests

customer_data = {
    "recency": 45,
    "frequency": 3,
    "monetary": 250.50,
    "avg_freight_value": 15.75,
    "tenure_days": 120,
    "avg_payment_value": 83.50,
    "avg_items_per_order": 2.3,
    "main_category": "electronics"
}

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json=customer_data
)

print(response.json())
# Output: {"churn_probability": 0.73, "risk_level": "high"}
```

## 🔬 Data Leakage Journey

### From 1.0 to 0.92: Debugging the "Too Perfect" Model

Initial models achieved a suspicious **1.0 AUC**—a major red flag. The debugging process involved three stages:

#### Stage 1: Obvious Leak
**Problem**: `recency` feature was directly used to define the target variable  
**Solution**: Removed from feature set

#### Stage 2: Subtle Leak
**Problem**: `tenure_days` was identical to `recency` for 95% of one-time buyers  
**Solution**: Removed from feature set

#### Stage 3: Technical Leak
**Problem**: `ColumnTransformer` using `remainder='passthrough'` leaked untransformed columns  
**Solution**: Changed to `remainder='drop'`

**Result**: Final AUC of 0.92—realistic, trustworthy, and production-ready.

> **Key Lesson**: In real-world data science, identifying *why* a model is perfect is more valuable than the perfect score itself.

## 📡 API Documentation

### Endpoints

#### `POST /predict`

Predicts churn probability for a single customer.

**Request Body**:
```json
{
  "recency": 45,
  "frequency": 3,
  "monetary": 250.50,
  "avg_freight_value": 15.75,
  "tenure_days": 120,
  "avg_payment_value": 83.50,
  "avg_items_per_order": 2.3,
  "main_category": "electronics"
}
```

**Response**:
```json
{
  "churn_probability": 0.73,
  "risk_level": "high"
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset provided by [Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- Built with scikit-learn, pandas, and FastAPI

---

**Made with ❤️ by [Your Name](https://github.com/yourusername)**
