# Laptop Price Prediction

This project predicts the price of laptops based on their specifications using machine learning and neural network models. It was developed for a coursework assignment focused on regression modeling, feature engineering, and model evaluation.

## Dataset

The dataset includes a variety of features for each laptop, such as:
- Company
- Product name
- RAM (e.g., 8GB)
- Weight (e.g., 1.37kg)
- Screen resolution
- CPU and GPU information
- Storage configuration (Memory)
- Operating system
- Price in Euros (target variable)

Source: Kaggle - Laptop Price Dataset

## Models Used

### Tree-Based Models
- Linear Regression (baseline)
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

### Neural Networks
- Basic dense feedforward neural network
- Deep neural network with embeddings and batch normalization

## Results (Validation Set)

| Model               | RMSE (€) | MAE (€) | R² Score |
|--------------------|----------|---------|----------|
| Linear Regression  | 357.60   | 221.56  | 0.8023   |
| Random Forest       | 358.79   | 210.09  | 0.8010   |
| Gradient Boosting   | 301.97   | 194.19  | 0.8590   |
| XGBoost             | 278.62   | 181.47  | 0.8800   |
| Deep Neural Network | 779.20   | 485.30  | 0.0612   |

XGBoost produced the best results across all evaluation metrics. Neural networks underperformed, likely due to the tabular structure of the dataset and relatively small sample size.

## Feature Engineering

Some columns were cleaned or modified for better model performance:
- `Ram` column was cleaned to remove the "GB" string and converted to an integer
- `Weight` column was cleaned to remove the "kg" suffix and converted to a float

These changes allowed the values to be treated as numerical rather than categorical.

Additional experiments with high-end laptop indicators (such as 4K screens, workstation types, or high RAM) were tested to help capture outliers.

## Visualizations

Each model includes:
- Actual vs Predicted price scatter plot (saved as a PNG)
- Feature importance bar chart for tree-based models

## Lessons Learned

- Tree-based models are very effective for structured/tabular data.
- Neural networks require careful tuning and more data to perform well in this domain.
- Feature importance helped identify key predictors like RAM, weight, and GPU type.
- All models had difficulty predicting very expensive laptops, likely due to limited examples or lack of detailed features describing premium components.

## Next Steps

- Improve outlier handling with separate models or better feature engineering
- Experiment with LightGBM or CatBoost for additional ensemble methods
- Tune neural networks further with regularization or residual connections
- Explore more advanced text preprocessing for CPU, GPU, and product fields

## Requirements

Install the necessary Python packages using:

```bash
pip install pandas numpy matplotlib scikit-learn xgboost tensorflow



