# DECISION-TREE-IMPLEMENTATION

**Company:** Codtech It Solutions

**Name:** Lalit Kumar

**Intern Id:** CT4MWZ33

**Domain:** Machine Learning

**Duration:** 16 Weeeks

**Mentor:** Neela Santhosh Kumar
📘 Overview
This project demonstrates a complete machine learning workflow—from data exploration and model building to deployment—using the Decision Tree Classifier on the popular Iris dataset. The project also includes a lightweight Flask web application that allows users to predict the species of iris flowers based on input features.

By combining scikit-learn, matplotlib, and Flask, this project offers both educational value and a hands-on demonstration of deploying ML models in real-world applications.

📊 Dataset Description
The Iris dataset is a classical dataset used in pattern recognition and machine learning. It contains 150 samples of iris flowers with the following features:

sepal length (cm)

sepal width (cm)

petal length (cm)

petal width (cm)

Each sample is labeled as one of three species:

Setosa

Versicolor

Virginica

✅ Project Steps
1. Data Loading and Exploration
We begin by importing the Iris dataset using sklearn.datasets.load_iris. Basic exploration such as printing feature and target names gives us insights into what the model will be learning and predicting.

python
Copy
Edit
print("Features:", iris.feature_names)
print("Target classes:", iris.target_names)
This confirms our 4 numerical features and 3 target classes.

2. Data Splitting
To evaluate the model's performance fairly, we split the dataset using train_test_split:

python
Copy
Edit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
This uses 70% for training and 30% for testing, ensuring reproducibility with random_state=42.

3. Model Training with Decision Tree
We initialize and train a Decision Tree Classifier with the entropy criterion (Information Gain) for better interpretability.

python
Copy
Edit
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)
4. Evaluation
After training, predictions are made and evaluated using classification_report and accuracy_score. In our case, the model achieves nearly 98% accuracy on unseen test data:

text
Copy
Edit
Classification Report:
               precision    recall  f1-score   support
        0       1.00      1.00      1.00        19
        1       0.93      1.00      0.96        13
        2       1.00      0.92      0.96        13
Accuracy Score: 0.98
5. Visualization
We visualize the trained decision tree using plot_tree() from sklearn.tree. This helps interpret the decision logic:

python
Copy
Edit
plt.figure(figsize=(24, 12))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
A textual representation of the tree is also generated with export_text().

🌐 Web Application with Flask
To make the model accessible, we built a simple Flask web application that:

Accepts 4 feature inputs from the user

Predicts the Iris species using the trained model

Displays results with styled HTML & CSS

Features:
Clean, responsive form

Error handling for invalid inputs

Clear output of predicted species

How it Works:
The Jupyter Notebook is converted to a .py script using nbconvert

The notebook code is executed once on server start, loading the model and dataset into memory

User inputs are taken through an HTML form and used to predict the class via clf.predict()

python
Copy
Edit
@app.route("/", methods=["GET", "POST"])
def predict():
    ...
You can start the web server with:

bash
Copy
Edit
python app.py
Then, visit http://127.0.0.1:5000 in your browser to access the interface.

📁 Project Structure
bash
Copy
Edit
├── app.py                 # Flask web app
├── Decision_Tree.ipynb    # Jupyter notebook (training and visualization)
├── static/                # (Optional) CSS or JS files
├── templates/             # (Optional) separate HTML templates
└── README.md              # Project description
💡 Future Improvements
Add input validation with JavaScript

Save trained model with joblib or pickle for faster loading

Extend to other classifiers (Random Forest, SVM)

Deploy with Docker or a cloud service like Heroku or AWS

🧠 Key Takeaways
Decision Trees are highly interpretable and powerful for classification tasks.

Scikit-learn provides an easy-to-use API for building and evaluating models.

Flask makes it easy to deploy machine learning models as web services.

This project demonstrates how to bridge the gap between ML modeling and real-world usability.

🛠 Requirements
Python 3.x

scikit-learn

Flask

matplotlib

numpy

pandas

nbformat, nbconvert (for reading the notebook programmatically)

You can install all dependencies using:

bash
Copy
Edit
pip install -r requirements.txt
🚀 Conclusion
This project offers a comprehensive introduction to building, interpreting, and deploying a machine learning model. It is perfect for beginners who want to see end-to-end development—from Jupyter notebooks to web applications.
#Output
![image](https://github.com/user-attachments/assets/5465e0c9-1df9-489a-af5d-628ed670bd50)
