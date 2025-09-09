**🎓 Student Performance Prediction App**
** 1. Objective of the Project**

The objective of this project is to predict a student's academic performance based on multiple factors such as study habits, attendance, past scores, and extracurricular activities.
This app helps educators, parents, and institutions to:

Identify students at risk of poor performance early.

Provide personalized learning plans for improvement.

Enhance decision-making in academic planning and interventions.

Real-World Problem Solved

Many students underperform because their weaknesses are identified too late. This solution predicts a student's Performance Index (numerical score) and Performance Category (Low, Medium, High) so that timely interventions can be made.

** 2. Data Source**

Source: The dataset used for this project was prepared for demonstration purposes, based on student academic performance features inspired by real educational data.

File Name: student_performance_with_unique_names.csv

Size: Approximately 100 rows × 6 columns

Features Included:

Name: Student’s name (unique)

Hours Studied: Average daily study hours

Previous Scores: Average score in previous exams

Attendance: Percentage of classes attended

Extracurricular Activities: Yes/No (participation in activities)

Sleep Hours: Average daily sleep hours

Performance Index: Final performance score (Target for Regression)

Performance Category: Derived category (Low, Medium, High)

 **3. Methodologies Used**

The project follows these steps:

Data Preprocessing:

Handling categorical features using Label Encoding.

Scaling numerical features for some models using StandardScaler.

Feature Selection: Removed unnecessary columns like Name for model training.

Model Training:

Split dataset into Train (80%) and Test (20%).

Evaluation: Used metrics like MAE, RMSE, R² Score for regression and Accuracy, Precision, Recall, F1-score for classification.

Visualization:

Confusion Matrix for classification models.

Performance Metrics comparison.

** 4. AI/ML Models Used**

We used a combination of Regression and Classification models:

For Performance Index (Regression):

Linear Regression – For predicting continuous performance score.

Random Forest Regressor – For better accuracy and handling nonlinear patterns.

For Performance Category (Classification):

Logistic Regression – For baseline classification.

Random Forest Classifier – For better accuracy.

Gradient Boosting Classifier – For robust and improved performance.

** 5. Predictions and Findings**

Predicted Performance Index: The app predicts the numerical performance score using Linear Regression and Random Forest, and provides an average prediction.

Predicted Performance Category: The app predicts whether the student falls into Low, Medium, or High category using three classification models.

Key Findings:

Hours Studied and Previous Scores are the strongest indicators of performance.

Students with high attendance and balanced sleep perform significantly better.

Participation in extracurricular activities has a moderate positive impact on performance.

** Features of the App**

✔ Predict student’s Performance Index
✔ Predict Performance Category (Low, Medium, High)
✔ Generate AI-Based Report with Strengths & Areas of Improvement
✔ Visualize Model Performance Metrics
✔ Downloadable Student Report

** How to Run the App**

Install dependencies:

pip install streamlit pandas numpy matplotlib seaborn scikit-learn


Run the app:

streamlit run app.py


Upload the dataset (or use the provided student_performance_with_unique_names.csv) in the same folder.

 **Future Enhancements**

Include more real-world features like family background, social media usage, etc.

Implement Deep Learning models for better accuracy.

Add real-time data input form for teachers and students.
