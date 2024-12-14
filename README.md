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

1. Clean email text data by removing special characters, stopwords, and more.
2. Use TF-IDF Vectorization to convert text data into numerical features for modeling.

-Handling Class Imbalance:

1. Address class imbalance with SMOTE to enhance model performance on minority classes.

- Model Training & Evaluation:

1. Train a Naive Bayes Classifier and evaluate the model using accuracy, confusion matrix, and advanced evaluation metrics.
2. Visualize model performance with ROC Curve and Precision-Recall Curve.

- Hyperparameter Tuning:

1. Fine-tune the model using GridSearchCV to achieve optimal results.
- Interactive Web App:

1. Predict whether a given email is spam or not via a simple Streamlit interface.
2. Save and view your prediction history in real-time.

## 📈 Demonstration of the Web App
[Link to Web App](https://spam-email-classification-jfwc792ttwneagun8lcdia.streamlit.app/)

https://github.com/user-attachments/assets/2df2fe6b-e45d-474a-ad49-ee31a9c7e6d2



## 🧑‍💻 Model Training & Evaluation
- Model Used:
1. Naive Bayes Classifier: A simple yet effective model for text classification tasks, especially for spam detection.
   
- Evaluation Metrics:
1. Accuracy: Measure of overall prediction correctness.
2. Confusion Matrix: Visual representation of true positives, false positives, true negatives, and false negatives.
3. ROC Curve: Plot the trade-off between the true positive rate and false positive rate.
4. Precision-Recall Curve: Plot showing precision vs. recall, useful for imbalanced datasets.
   
- Cross-Validation & Hyperparameter Tuning:
1. We perform cross-validation to validate model performance across different data splits.
2. GridSearchCV is used for hyperparameter tuning to find the optimal parameters.

## 📈 Comparison: With vs. Without Resampling
One of the key aspects of this project is the handling of class imbalance. The dataset contains significantly more non-spam emails than spam emails, which could lead to a biased model that predicts only the majority class (non-spam) with high accuracy but performs poorly on spam classification.
- ![image](https://github.com/user-attachments/assets/7f4d9251-290d-4954-9805-c2c3a61ca8f6)
- ![image](https://github.com/user-attachments/assets/97796e3f-9353-45e7-9b1d-3c07306ae413)

 As you can see, while the accuracy is slightly lower with resampling, the precision and recall for spam emails are significantly improved. This shows that the model is now better equipped to correctly classify spam emails without being biased towards the majority class.


## 🌟 Key Technologies
1. Python: The programming language for this entire project.
2. Scikit-learn: For building the machine learning model, evaluating performance, and performing cross-validation.
3. Pandas & NumPy: For data manipulation and processing.
4. Plotly & Matplotlib: For interactive and static visualizations.
5. Seaborn: For creating beautiful and informative statistical graphics.
6. WordCloud: For generating word clouds of common words in spam and non-spam emails.
7. SMOTE: For handling class imbalance in the dataset.
8. Streamlit: For creating the web interface for real-time predictions.
9. Joblib: To save and load the trained machine learning model for deployment.

    
## 🎯 How It Works
- User Input: The user inputs the email text in the web app.
- Text Preprocessing: The email text undergoes preprocessing (removal of special characters, stopwords, etc.).
- Prediction: The processed text is fed into the trained Naive Bayes model.
- Classification: The model classifies the email as Spam or Not Spam based on learned patterns.
- Feedback: The classification result is displayed to the user in real-time.

## 🤝 Contributing
We welcome contributions to improve this project! Whether it's bug fixes, new features, or enhancements to the existing code, feel free to fork the repository and submit a pull request.
