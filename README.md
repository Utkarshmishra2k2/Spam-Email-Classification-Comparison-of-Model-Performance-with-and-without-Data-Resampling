# Spam Email Classification with Machine Learning
Welcome to the Spam Email Classification project! 🚀 This project uses machine learning to automatically classify emails as Spam or Not Spam. With a blend of data exploration, text preprocessing, model training, and deployment via a web application, this repository showcases the entire pipeline for solving a common NLP problem.

Whether you're a beginner or a seasoned ML enthusiast, this project will give you hands-on experience with data cleaning, feature extraction, model training, and building an interactive web application using Streamlit. 💻✨

## 🔍 Project Overview
This project is a comprehensive end-to-end solution for spam email detection. It leverages the following:

- Data Exploration: Insightful visualizations to understand the dataset.
- Text Preprocessing: Techniques to clean and prepare the data for model training.
- Feature Engineering: Use of TF-IDF for text vectorization.
- Modeling: Training a Naive Bayes classifier to classify emails as spam or not.
- Handling Imbalanced Data: Using SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.
- Model Evaluation: Metrics like accuracy, confusion matrix, ROC Curve, and Precision-Recall Curve.
- Hyperparameter Tuning: Using GridSearchCV to find the best model parameters.
- Real-time Prediction: An interactive Streamlit app to predict spam emails on the go!


## 🛠 Key Features
- Data Exploration & Visualization:
1. Analyze the distribution of spam vs. non-spam emails.
2. Explore text length distribution and identify key patterns.
3. Visualize frequent words in both spam and non-spam emails using WordCloud.

- Text Preprocessing & Feature Extraction:

Clean email text data by removing special characters, stopwords, and more.
Use TF-IDF Vectorization to convert text data into numerical features for modeling.
Handling Class Imbalance:

Address class imbalance with SMOTE to enhance model performance on minority classes.
Model Training & Evaluation:

Train a Naive Bayes Classifier and evaluate the model using accuracy, confusion matrix, and advanced evaluation metrics.
Visualize model performance with ROC Curve and Precision-Recall Curve.
Hyperparameter Tuning:

Fine-tune the model using GridSearchCV to achieve optimal results.
Interactive Web App:

Predict whether a given email is spam or not via a simple Streamlit interface.
Save and view your prediction history in real-time.
🚀 How to Run the Project Locally
Clone the Repository
bash
Copy code
git clone https://github.com/your-username/spam-email-classification.git
cd spam-email-classification
Install Dependencies
Install the required Python packages using pip:

bash
Copy code
pip install -r requirements.txt
Run the Streamlit Web App
To run the interactive web application, use the following command:

bash
Copy code
streamlit run app.py
This will launch the web app in your browser where you can start classifying emails as spam or not!

🧑‍💻 Model Training & Evaluation
Model Used:
Naive Bayes Classifier: A simple yet effective model for text classification tasks, especially for spam detection.
Evaluation Metrics:
Accuracy: Measure of overall prediction correctness.
Confusion Matrix: Visual representation of true positives, false positives, true negatives, and false negatives.
ROC Curve: Plot the trade-off between the true positive rate and false positive rate.
Precision-Recall Curve: Plot showing precision vs. recall, useful for imbalanced datasets.
Cross-Validation & Hyperparameter Tuning:
We perform cross-validation to validate model performance across different data splits.
GridSearchCV is used for hyperparameter tuning to find the optimal parameters.
🌟 Key Technologies
Python: The programming language for this entire project.
Scikit-learn: For building the machine learning model, evaluating performance, and performing cross-validation.
Pandas & NumPy: For data manipulation and processing.
Plotly & Matplotlib: For interactive and static visualizations.
Seaborn: For creating beautiful and informative statistical graphics.
WordCloud: For generating word clouds of common words in spam and non-spam emails.
SMOTE: For handling class imbalance in the dataset.
Streamlit: For creating the web interface for real-time predictions.
Joblib: To save and load the trained machine learning model for deployment.
