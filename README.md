# Predicting Student Health Risk

A machine learning project developed to predict a student's health condition using lifestyle, behavioural, and health-related information. The system classifies students into one of three health-risk categories:

- **At-risk**
- **Fit**
- **Unhealthy**

The project compares three computational intelligence models and provides both a **Streamlit web application** and a **FastAPI prediction API**.

## Project Overview

Student health can be influenced by factors such as sleep, physical activity, stress, smoking, alcohol use, and daily habits. This project uses machine learning to identify patterns in these factors and predict the likely health condition of a student.

The dataset contains approximately **690,000 training records**, thirteen input features, and a three-class target variable. Exploratory data analysis showed that the target classes were imbalanced, with the **at-risk** class containing most of the records. Therefore, model performance was evaluated using accuracy together with macro precision, macro recall, and macro F1-score.

## Models Used

The following models were trained and compared:

1. **XGBoost Classifier**
2. **Neural Network**
3. **Random Forest Classifier**

### Model Performance

| Model | Macro F1-score |
|---|---:|
| XGBoost | 0.9076 |
| Neural Network | 0.9000 |
| Random Forest | 0.8750 |

XGBoost produced the best overall result, with approximately **96.6% test accuracy** and the highest macro F1-score.

## Main Features

- Exploratory data analysis and chart generation
- Missing-value handling
- Numerical and categorical feature preprocessing
- Training and comparison of three machine learning models
- Saved trained models using Joblib/Pickle
- Streamlit-based user interface
- FastAPI prediction endpoint
- Input validation for prediction requests
- Reusable preprocessing and prediction pipeline

## Project Structure

```text
Predicting Student Health Risk/
│
├── comparison_models/
│   ├── load_and_predict_example.py
│   ├── neural_network_model.pkl
│   ├── random_forest_model.pkl
│   └── xgboost_model.pkl
│
├── model_artifacts/
│   ├── preprocessing.pkl
│   ├── neural_network_model.pkl
│   ├── random_forest_model.pkl
│   └── xgboost_model.pkl
│
├── notebooks/
│   ├── eda_plots/
│   ├── generate_eda_charts.py
│   ├── neural_network_model.ipynb
│   ├── random_forest_model.ipynb
│   ├── xgboost_model.ipynb
│   ├── train.csv
│   └── test.csv
│
├── webapp/
│   ├── api.py
│   ├── streamlit_app.py
│   ├── requirements.txt
│   ├── preprocessing.pkl
│   └── trained model files
│
├── train_model.py
├── .gitignore
└── README.md
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Streamlit
- FastAPI
- Uvicorn
- Matplotlib
- Jupyter Notebook

## Installation

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd "Predicting Student Health Risk"
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source venv/bin/activate
```

### 3. Install the required packages

```bash
pip install -r webapp/requirements.txt
pip install fastapi uvicorn
```

## Running the Streamlit Application

From the main project directory, run:

```bash
streamlit run webapp/streamlit_app.py
```

Streamlit will display a local address, normally:

```text
http://localhost:8501
```

Open the address in a web browser, enter the student information, select a model, and submit the form to receive a predicted health condition.

## Running the FastAPI Application

From the main project directory, run:

```bash
uvicorn webapp.api:app --reload
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation can be opened at:

```text
http://127.0.0.1:8000/docs
```

## Training the Models

To run the main model-training script:

```bash
python train_model.py
```

The notebooks inside the `notebooks` directory can also be used to review the separate training and evaluation process for each model.

## Exploratory Data Analysis

The EDA process examines:

- Target-class distribution
- Missing values
- Numerical feature relationships
- Categorical feature relationships
- Correlation between numerical variables

The generated charts are stored in:

```text
notebooks/eda_plots/
```

## Evaluation

The models were evaluated using:

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Classification report
- Confusion matrix

Macro-averaged measures were important because the dataset had a strong class imbalance. Accuracy alone could give a misleading result by favouring the majority class.

## Best Model

XGBoost was selected as the strongest model because it achieved the best overall balance between accuracy, precision, recall, and macro F1-score. It performed very well on the majority class while also producing strong predictions for the smaller fit and unhealthy classes.

## Important Note

This application is an academic machine learning project. Its predictions are intended for demonstration and research purposes only. It must not be treated as professional medical advice or used as a replacement for a qualified healthcare professional.

## Author

**Sunera Nawod**

Final-year Computational Intelligence project.

## License

This project is provided for educational and academic use.
