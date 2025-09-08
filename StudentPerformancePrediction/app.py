import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Student Performance Prediction App",
    page_icon="👨‍🎓",
    layout="wide"
)

# --- Helper Functions ---
def performance_category(value):
    """Categorizes the performance index."""
    if value < 50:
        return "Low"
    elif 50 <= value < 75:
        return "Medium"
    else:
        return "High"

@st.cache_resource
def load_and_train_models():
    """Loads data, trains models, and returns them along with the full dataframe."""
    try:
        df = pd.read_csv('student_performance_with_names.csv')
    except FileNotFoundError:
        st.error("Error: 'student_performance_with_names.csv' not found. Please place the file in the same directory.")
        st.stop()
    
    # Preprocessing
    df['Extracurricular Activities'] = LabelEncoder().fit_transform(df['Extracurricular Activities'])
    df['Performance Category'] = df['Performance Index'].apply(performance_category)

    # Features and Target for Regression
    X_reg = df.drop(['Performance Index', 'Name', 'Performance Category'], axis=1)
    y_reg = df['Performance Index']

    # Features and Target for Classification
    X_clf = df.drop(['Performance Category', 'Name', 'Performance Index'], axis=1)
    y_clf = df['Performance Category']
    
    # Train-Test Split
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

    # Train Models
    lin_reg = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    lin_reg.fit(X_train_r, y_train_r)

    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train_r, y_train_r)
    
    log_reg = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])
    log_reg.fit(X_train_c, y_train_c)

    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train_c, y_train_c)

    gb_clf = GradientBoostingClassifier()
    gb_clf.fit(X_train_c, y_train_c)
    
    return lin_reg, rf_reg, log_reg, rf_clf, gb_clf, X_test_r, y_test_r, X_test_c, y_test_c, df

# --- Main Streamlit App ---
st.title("🎓 Student Performance Prediction")
st.markdown("This app predicts a student's performance index and category based on various input features.")

# Load models and data
with st.spinner('Training models... This may take a moment.'):
    lin_reg, rf_reg, log_reg, rf_clf, gb_clf, X_test_r, y_test_r, X_test_c, y_test_c, df = load_and_train_models()
st.success("✅ Models trained successfully!")

# Sidebar input
st.sidebar.header("Student Information")
student_name = st.sidebar.text_input("Enter Student's Name")
generate_report_button = st.sidebar.button("Generate Report")

# --- Prediction and Report ---
if generate_report_button:
    if student_name:
        student_data = df[df['Name'].str.lower() == student_name.lower()]
        
        if not student_data.empty:
            st.header("📋 Student Information")
            st.dataframe(student_data)

            st.header("📊 Prediction & Analysis")

            # Regression Predictions
            st.subheader("Performance Index Prediction (Regression)")
            input_data = student_data.drop(['Performance Index', 'Name', 'Performance Category'], axis=1)
            lin_reg_pred = lin_reg.predict(input_data)[0]
            rf_reg_pred = rf_reg.predict(input_data)[0]
            avg_reg_pred = (lin_reg_pred + rf_reg_pred) / 2

            st.write(f"**Linear Regression Prediction:** {lin_reg_pred:.2f}")
            st.write(f"**Random Forest Regressor Prediction:** {rf_reg_pred:.2f}")
            st.write(f"**Average Predicted Index:** {avg_reg_pred:.2f}")

            # Classification Predictions
            st.subheader("Performance Category Prediction (Classification)")
            log_reg_pred = log_reg.predict(input_data)[0]
            rf_clf_pred = rf_clf.predict(input_data)[0]
            gb_clf_pred = gb_clf.predict(input_data)[0]

            st.write(f"**Logistic Regression Prediction:** {log_reg_pred}")
            st.write(f"**Random Forest Prediction:** {rf_clf_pred}")
            st.write(f"**Gradient Boosting Prediction:** {gb_clf_pred}")

            # --- AI Report ---
            st.header("📝 AI-Generated Report")
            predicted_performance_index = avg_reg_pred
            predicted_performance_category = performance_category(avg_reg_pred)

            strengths = "Shows consistent effort in studies."
            weaknesses = "Needs to improve on time management and test performance."
            areas_of_improvement = "Focus on weak topics, practice more past papers, and maintain study discipline."

            report_text = f"""
============================
📌 Student Performance Report
============================

**Student Name:** {student_data.iloc[0]['Name']}

**Actual Performance Index:** {student_data.iloc[0]['Performance Index']}
**Predicted Performance Index:** {predicted_performance_index:.2f}
**Predicted Category:** {predicted_performance_category}

----------------------------
🔹 Strengths
{strengths}

🔸 Weaknesses
{weaknesses}

⚡ Areas of Improvement
{areas_of_improvement}
----------------------------
"""

            st.markdown(report_text)

            # Download button
            st.download_button(
                label="📥 Download Report",
                data=report_text,
                file_name=f"{student_name.replace(' ', '_')}_report.txt",
                mime="text/plain"
            )
            
        else:
            st.warning(f"⚠️ Student '{student_name}' not found in the database. Please check the spelling.")
    else:
        st.info("ℹ️ Please enter a student's name to generate a report.")

# --- Model Performance Metrics ---
st.header("📉 Model Performance Metrics")
tab1, tab2 = st.tabs(["Regression Metrics", "Classification Metrics"])

with tab1:
    st.subheader("Regression Models")

    y_pred_lin = lin_reg.predict(X_test_r)
    st.markdown("##### Linear Regression")
    st.write(f"**MAE:** {mean_absolute_error(y_test_r, y_pred_lin):.4f}")
    st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test_r, y_pred_lin)):.4f}")
    st.write(f"**R2 Score:** {r2_score(y_test_r, y_pred_lin):.4f}")

    y_pred_rf = rf_reg.predict(X_test_r)
    st.markdown("##### Random Forest Regressor")
    st.write(f"**MAE:** {mean_absolute_error(y_test_r, y_pred_rf):.4f}")
    st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test_r, y_pred_rf)):.4f}")
    st.write(f"**R2 Score:** {r2_score(y_test_r, y_pred_rf):.4f}")

with tab2:
    st.subheader("Classification Models")

    y_pred_log = log_reg.predict(X_test_c)
    st.markdown("##### Logistic Regression")
    st.text(classification_report(y_test_c, y_pred_log))

    y_pred_rf_clf = rf_clf.predict(X_test_c)
    st.markdown("##### Random Forest Classifier")
    st.text(classification_report(y_test_c, y_pred_rf_clf))

    y_pred_gb = gb_clf.predict(X_test_c)
    st.markdown("##### Gradient Boosting Classifier")
    st.text(classification_report(y_test_c, y_pred_gb))

    st.subheader("Confusion Matrices")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sns.heatmap(confusion_matrix(y_test_c, y_pred_log), annot=True, fmt='d', cmap="Blues", ax=axes[0])
    axes[0].set_title("Logistic Regression")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    sns.heatmap(confusion_matrix(y_test_c, y_pred_rf_clf), annot=True, fmt='d', cmap="Greens", ax=axes[1])
    axes[1].set_title("Random Forest")
    axes[1].set_xlabel("Predicted")

    sns.heatmap(confusion_matrix(y_test_c, y_pred_gb), annot=True, fmt='d', cmap="Oranges", ax=axes[2])
    axes[2].set_title("Gradient Boosting")
    axes[2].set_xlabel("Predicted")

    plt.tight_layout()
    st.pyplot(fig)
