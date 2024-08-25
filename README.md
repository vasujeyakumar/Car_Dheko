# Car Price Prediction using Gradient Boosting Algorithm

## 1. Problem Statement
Predict the price of a car based on various features such as mileage, make, model, engine size, and year using a gradient boosting algorithm model.

## 2. Objective
To develop a gradient boosting regression model that predicts car prices based on historical sales data and car features.

## 3. Data Processing and Extraction

### 3.1 Dataset Overview
- **Source**: Collected from various car listing datasets, including CarDekho
- **Format**: CSV format with numerical and categorical features related to car specifications and sales.

### 3.2 Features
- **Numerical Features**:
  - `Registeration Year`: Year of car Registeration.
  - `length`: Length of car.
  - `width`: Width of car.
  
- **Categorical Features**:
  - `state`: city  (e.g., chennai, bangalore).
  - `Model`: Specific model (e.g., Corolla, Mustang).

### 3.3 Target Variable
- **Price**: The selling price of the car.

## 4. Data Preparation

### 4.1 Converting Lists and Dictionaries to DataFrames
Data was transformed from lists and dictionaries into DataFrames for better manipulation.

### 4.2 Data Cleaning
- **Handling Missing Values**: 
  - Filled missing values for numerical features with median values and for categorical features with mode.

### 4.3 Feature Engineering
- **Categorical Encoding**: Applied one-hot encoding to categorical features.
- **Data Type Conversion**: Cleaned and converted data types for correct processing.

### 4.4 Data Splitting
- Split the dataset into 80% training and 20% testing.

## 5. Model Development

### 5.1 Model Selection
Gradient Boosting Algorithm was chosen for its capability to handle complex feature interactions.

### 5.2 Model Training
The model was trained using `GradientBoostingRegressor` with specific hyperparameters.

### 5.3 Hyperparameters
- `n_estimators=300`
- `random_state=42`
- `learning_rate=0.1`
- `max_depth=4`
- `min_samples_leaf=2`
- `min_samples_split=2`
-  'subsample: 0.8'

## 6. Model Evaluation

### 6.1 Performance Metrics
- **Mean Absolute Error (MAE)**: 65919.34
- **Mean Squared Error (MSE)**: 9778261071.31
- **R-squared (R²)**: 0.90

### 6.2 Key Features
- Body Type
- Model Year
- Transmission
- Mileage

These features are significant in predicting car prices.

## 7. Model Deployment

### 7.1 Deployment Method
Deployed as a Streamlit web app to allow users to input car features and receive price predictions.

## 8. Future Improvements
- Explore hyperparameter tuning using GridSearchCV for further optimization.
- Evaluate additional models and techniques for improved performance.

## 9. References
- **Dataset Source**: CarDekho
- **Scikit-learn Documentation**: [Scikit-learn](https://scikit-learn.org/stable/)

## Appendix
- **Model Artifacts**: Trained model and preprocessor pipeline are saved as `.pkl` files.
- **Code Repository**: [GitHub Repository](https://github.com/your-repo-link)

## Conclusion
The gradient boosting model effectively predicts car prices, demonstrating robust performance with an R² value of 0.91.
The model highlights the importance of features like Body Type and Model Year, which significantly influence pricing. 
Future work includes refining the model through hyperparameter tuning and exploring additional algorithms to enhance predictive accuracy.
The deployment as a Streamlit app provides a user-friendly interface for real-time price estimation, making the model accessible to potential users and stakeholders.
