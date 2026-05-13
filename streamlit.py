import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Page config
st.set_page_config(page_title="Decision Tree Regressor", layout="wide")
st.title("Decision Tree Regressor - IRIS Dataset")

# Load and prepare data
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv('IRIS.csv')
    
    # Prepare features and target
    x = df.drop('petal_length', axis=1)
    y = df['petal_length']
    
    # Encode categorical variables
    categorical_cols = x.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        x[col] = le.fit_transform(x[col])
        label_encoders[col] = le
    
    return df, x, y, label_encoders

@st.cache_resource
def train_model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = DecisionTreeRegressor(criterion='squared_error', max_depth=6, 
                                  max_features='log2', splitter='best')
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    return model, metrics, x.columns.tolist()

# Load data
df, x, y, label_encoders = load_and_prepare_data()
model, metrics, feature_names = train_model(x, y)

# Display metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Mean Squared Error", f"{metrics['mse']:.4f}")
with col2:
    st.metric("Mean Absolute Error", f"{metrics['mae']:.4f}")
with col3:
    st.metric("R² Score", f"{metrics['r2']:.4f}")

st.divider()

# User input section
st.subheader("Make a Prediction")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length (cm)", float(df['sepal_length'].min()), 
                             float(df['sepal_length'].max()), 5.8)
    sepal_width = st.slider("Sepal Width (cm)", float(df['sepal_width'].min()), 
                            float(df['sepal_width'].max()), 3.0)

with col2:
    petal_width = st.slider("Petal Width (cm)", float(df['petal_width'].min()), 
                            float(df['petal_width'].max()), 1.2)
    species = st.selectbox("Species", ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])

# Encode species
species_encoded = label_encoders['species'].transform([species])[0]

# Make prediction
input_data = np.array([[sepal_length, sepal_width, petal_width, species_encoded]])
prediction = model.predict(input_data)[0]

# Display prediction
st.divider()
st.success(f"### Predicted Petal Length: **{prediction:.4f} cm**")

# Display sample data
with st.expander("View Dataset Sample"):
    st.dataframe(df.head(10))
