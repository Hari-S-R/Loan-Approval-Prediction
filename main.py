import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("LoanApprovalPrediction.csv")

data.drop(['Loan_ID'],axis=1,inplace=True)

label_encoder={}
for i in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']:
      le = LabelEncoder()
      data[i] = le.fit_transform(data[i])
      label_encoder[i] = le

# Find and fill the missing values
for i in data.columns:
  data[i] = data[i].fillna(data[i].mean()) 
  
print(data.isna().sum())

# Splitting data into x,y
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

#train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42)

# Making predictions on the training set
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
acc = accuracy_score(y_test, y_pred)
cr = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Predict for a new applicant
sample = [[1, 1, 0, 0, 0, 5000, 0.0, 128, 360.0, 1.0, 2]]  # dummy values
prediction = model.predict(sample)
print("\nPrediction for new applicant:", "Loan Approved" if prediction == 1 else "Loan Rejected")

# Streamlit code
# Title
st.title("Loan Approval Prediction App")

# Sidebar content
st.sidebar.header("📊 Model Evaluation")

# Show classification report in sidebar
st.sidebar.subheader("🔍 Classification Report")
st.sidebar.text(cr)

# Confusion matrix plot
st.sidebar.subheader("📈 Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Rejected", "Approved"], yticklabels=["Rejected", "Approved"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.sidebar.pyplot(fig)

# User input form
st.header("📋 Enter Applicant Details:")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0.0)
loan_term = st.number_input("Loan Amount Term (in days)", min_value=0.0, value=360.0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# Predict button
if st.button("Predict Loan Status"):
    # Encode categorical inputs
    input_data = [
        label_encoder['Gender'].transform([gender])[0],
        label_encoder['Married'].transform([married])[0],
        int(dependents.replace("3+", "3")),
        label_encoder['Education'].transform([education])[0],
        label_encoder['Self_Employed'].transform([self_employed])[0],
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        label_encoder['Property_Area'].transform([property_area])[0]
    ]

    prediction = model.predict([input_data])[0]
    result = label_encoder['Loan_Status'].inverse_transform([prediction])[0]

    st.success(f"Loan Status: {'Approved ✅' if result == 'Y' else 'Rejected ❌'}")