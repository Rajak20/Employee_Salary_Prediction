# 💼 Employee Salary Prediction

## 📌 Project Overview

Employee Salary Prediction is a Machine Learning project that predicts an employee's salary based on various factors such as age, work experience, education level, job role, city tier, performance score, number of skills, and remote work status.

This project uses **Linear Regression** to estimate employee salaries and **StandardScaler** for feature scaling. A user-friendly web application is developed using Streamlit to make salary predictions easily.

## 🚀 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Pickle
- Linear Regression
- StandardScaler

## 📊 Input Features

The model uses the following eight input features:

| Feature | Description |
|---|---|
| age | Employee's age |
| years_experience | Years of work experience |
| education_level | Education level (numerically encoded) |
| job_role | Employee's job role (numerically encoded) |
| city_tier | City classification (numerically encoded) |
| performance_score | Employee performance score |
| num_skills | Number of skills possessed |
| remote_work | Remote work status (0 or 1) |

## 🤖 Machine Learning Algorithm

**Linear Regression**

Linear Regression is a supervised machine learning algorithm used to predict continuous numerical values. In this project, it predicts employee salaries based on the provided input features.

### Data Preprocessing

- Feature scaling using StandardScaler.
- Transformation of input features before prediction.
- Consistent feature ordering between training and prediction.

## 🌐 Streamlit Web Application

The Streamlit application provides an interactive interface where users can:

- Enter employee details.
- Provide education and job role encodings.
- Select remote work status.
- Predict employee salary.
- View the predicted salary instantly.

## 📂 Project Structure

```text
Employee-Salary-Prediction/
│
├── app.py
├── salary_model.pkl
├── scaler (6).pkl
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

**1. Clone the repository**

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

**2. Navigate to the project directory**

```bash
cd Employee-Salary-Prediction
```

**3. Install the required libraries**

```bash
pip install -r requirements.txt
```

**4. Run the Streamlit application**

```bash
streamlit run app.py
```

## 📈 Prediction Workflow

1. Collect employee details.
2. Prepare the input features.
3. Apply StandardScaler to transform the features.
4. Pass the transformed features to the trained Linear Regression model.
5. Generate the predicted employee salary.
6. Display the prediction through the Streamlit interface.

## 🎯 Project Objective

The main objective of this project is to demonstrate how Machine Learning can be used to estimate employee salaries based on multiple influencing factors through an interactive web application.

## 👨‍💻 Author

**Rajak**

GitHub: [Rajak20](https://github.com/Rajak20)
