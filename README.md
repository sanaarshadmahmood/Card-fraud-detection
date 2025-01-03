# Credit Card Fraud Detection

## Overview
This project is designed to detect fraudulent transactions using machine learning techniques. The dataset used is sourced from Hugging Face and contains features such as `time_elapsed`, `amt`, `lat`, `long`, and `is_fraud`. The project demonstrates various steps of data preprocessing, exploratory data analysis (EDA), and model evaluation to identify fraudulent transactions effectively.

## Dataset
- **Source**: Hugging Face ([tanzuhuggingface/creditcardfraudtraining](https://huggingface.co/datasets))
- **Features**: 
  - `time_elapsed`: Time elapsed from the first transaction.
  - `amt`: Transaction amount.
  - `lat` and `long`: Latitude and longitude of the transaction location.
  - `is_fraud`: Indicates whether the transaction is fraudulent (1) or not (0).

## Key Steps
1. **Install Required Libraries**
   - Install necessary libraries, including Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, and XGBoost.

2. **Import Libraries**
   - Essential imports for data manipulation, visualization, and machine learning.

3. **Data Preprocessing**
   - Handle missing values, normalize features, and encode categorical variables.

4. **Exploratory Data Analysis (EDA)**
   - Visualizations include:
     - Count plot for class distribution.
     - Distribution and correlation analysis.
     - Scatter and pair plots for feature exploration.

5. **Model Training and Evaluation**
   - Models used: Logistic Regression, XGBoost, Decision Tree, Random Forest, MLPClassifier.
   - Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Training Time.
   - Visualizations: Confusion matrices and ROC curves.

## Results
- The **XGBoost Classifier** achieved the best performance with:
  - **Accuracy**: 99%
  - Excellent metrics across precision, recall, F1-score, and ROC-AUC.

## How to Use the Code
1. Clone the repository:
   ```bash
   git clone https://github.com/sanaarshadmahmood/Card-fraud-detection.git
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:
   ```bash
   jupyter notebook Credit_Card.ipynb
   ```

## Dependencies
- Python 3.8+
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost

## Outputs
- Confusion matrices and ROC curves are generated for each model.
- Comprehensive evaluation of machine learning models.

## Contact
For any queries or suggestions, please feel free to reach out to mssanaarshad@gmail.com.

## License
This project is licensed under the MIT License.
