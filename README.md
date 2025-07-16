# 🌸 Decision Tree Classifier on Iris Dataset with Flask Web Interface

## 📘 Overview

This project demonstrates a complete machine learning workflow—from data exploration and model building to deployment—using the **Decision Tree Classifier** on the popular **Iris dataset**. The project also includes a lightweight **Flask web application** that allows users to predict the species of iris flowers based on input features.

By combining **scikit-learn**, **matplotlib**, and **Flask**, this project offers both educational value and a hands-on demonstration of deploying ML models in real-world applications.

---

## 📊 Dataset Description

The **Iris dataset** is a classical dataset used in pattern recognition and machine learning. It contains 150 samples of iris flowers with the following features:

* **sepal length (cm)**
* **sepal width (cm)**
* **petal length (cm)**
* **petal width (cm)**

Each sample is labeled as one of three species:

* **Setosa**
* **Versicolor**
* **Virginica**

---

## ✅ Project Steps

### 1. **Data Loading and Exploration**

We begin by importing the Iris dataset using `sklearn.datasets.load_iris`. Basic exploration such as printing feature and target names gives us insights into what the model will be learning and predicting.

### 2. **Data Splitting**

To evaluate the model's performance fairly, we split the dataset using `train_test_split`, using 70% for training and 30% for testing, with a fixed `random_state` for reproducibility.

### 3. **Model Training with Decision Tree**

We initialize and train a Decision Tree Classifier with the `entropy` criterion (Information Gain). This helps in gaining better interpretability.

### 4. **Evaluation**

After training, predictions are made and evaluated using `classification_report` and `accuracy_score`. The model achieves nearly **98% accuracy** on test data.

### 5. **Visualization**

We visualize the trained decision tree using `plot_tree()` and provide a textual representation with `export_text()` to interpret the decision paths.

---

## 🌐 Web Application with Flask

To make the model accessible, we built a simple **Flask web application** that:

* Accepts 4 feature inputs from the user
* Predicts the Iris species using the trained model
* Displays results with styled HTML & CSS

### Features:

* Clean, responsive form
* Error handling for invalid inputs
* Clear output of predicted species

### How it Works:

* The Jupyter Notebook is converted to Python code using `nbconvert`
* The model and data are loaded and executed once when the server starts
* User inputs are submitted via an HTML form and predicted with `clf.predict()`

Start the server using:

```bash
python app.py
```

Then, visit `http://127.0.0.1:5000` in your browser.

---

## 📁 Project Structure

```bash
├── app.py                 # Flask web app
├── Decision_Tree.ipynb    # Jupyter notebook (training and visualization)
├── static/                # (Optional) CSS or JS files
├── templates/             # (Optional) separate HTML templates
└── README.md              # Project description
```

---

## 💡 Future Improvements

* Add input validation with JavaScript
* Save trained model with `joblib` or `pickle` for faster loading
* Extend to other classifiers (Random Forest, SVM)
* Deploy with Docker or cloud services like Heroku or AWS

---

## 🧠 Key Takeaways

* **Decision Trees** are highly interpretable and powerful for classification tasks.
* **Scikit-learn** provides an easy-to-use API for building and evaluating models.
* **Flask** makes it easy to deploy machine learning models as web services.
* This project bridges the gap between model development and practical deployment.

---

## 🛠 Requirements

* Python 3.x
* scikit-learn
* Flask
* matplotlib
* numpy
* pandas
* nbformat, nbconvert

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🚀 Conclusion

This project offers a comprehensive introduction to building, interpreting, and deploying a machine learning model. It is ideal for beginners who want to explore end-to-end ML development—from Jupyter notebooks to user-friendly web applications.

# Output
![image](https://github.com/user-attachments/assets/5465e0c9-1df9-489a-af5d-628ed670bd50)
