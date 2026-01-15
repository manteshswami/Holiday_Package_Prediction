# Holiday Package Purchase Prediction

A machine learning system designed to predict whether a customer will purchase a holiday package, enabling data-driven marketing for Trips & Travel.Com.

The model helps the company reduce acquisition cost by identifying high-probability customers instead of running expensive random campaigns.

---
## How to Clone and Run
*Clone the repository:*

```bash
git clone https://github.com/manteshswami/Holiday_Package_Prediction.git
cd Holiday_Package_Prediction
---

## Project Structure

Holiday_Package_Prediction/
│
├── holiday_package_prediction.py
├── Holiday_Package_Prediction.ipynb
├── Travel.csv
├── holiday_package_model.pkl
├── preprocessor.pkl
├── README.md

---
## Business Objective

Trips & Travel.Com offers multiple holiday packages such as **Basic, Standard, Deluxe, Super Deluxe, and King**.  
Historical data shows that only about **18% of customers** convert, while the marketing cost remains high due to non-targeted outreach.

This project builds a predictive model that estimates the probability of a customer purchasing a newly introduced **Wellness Tourism Package**, allowing the business to focus marketing efforts on customers most likely to convert.

---

## Dataset

The dataset contains customer demographics, travel history, and income-related features collected from previous campaigns.

- Rows: 4,888  
- Features: 20  
- Target variable: `ProdTaken`  
  - `1` → Package purchased  
  - `0` → Not purchased  

---

## Data Processing

The following data preparation steps were applied:

- Standardized inconsistent labels  
  - `Fe Male` → `Female`  
  - `Single` → `Unmarried`
- Missing value handling  
  - Numerical features → Median imputation  
  - Categorical features → Mode imputation  
- Removed non-predictive column: `CustomerID`
- Feature engineering  
  - Created `TotalVisiting` = `NumberOfChildrenVisiting + NumberOfPersonVisiting`

---

## Feature Transformation

A unified preprocessing pipeline was built using `ColumnTransformer`:

- Categorical features → One-Hot Encoding (drop first)
- Numerical features → Standard Scaling

This ensures consistent feature transformation during both training and inference.

---

## Model

A **Random Forest Classifier** was selected due to its robustness and ability to capture non-linear relationships.

The final model was trained using optimized hyperparameters:

- `n_estimators = 500`
- `max_features = 8`
- `min_samples_split = 2`
- `max_depth = None`

---

## Model Evaluation

The trained model delivers strong predictive performance on unseen data, with high accuracy, precision, recall, and ROC-AUC score.  
All evaluation metrics are printed when running the training script.

---

## Deployment-Ready Artifacts

After training, the following files are generated:
- holiday_package_model.pkl
- preprocessor.pkl
