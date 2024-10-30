import streamlit as st
import pickle
import numpy as np

# Load the model from the file
with open('modelGaussianNB.pkl', 'rb') as f:
    modelGaussianNB = pickle.load(f)

with open('modelKNeighborsClassifier.pkl', 'rb') as f:
    modelKNeighborsClassifier = pickle.load(f)

with open('modelPerceptron.pkl', 'rb') as f:
    modelPerceptron = pickle.load(f)

with open('modelSVM.pkl', 'rb') as f:
    modelSVM = pickle.load(f)

# Streamlit app
st.title("BankNotes App")

# Get inputs
variance = st.text_input("Enter variance:")
skewness = st.text_input("Enter skewness:")
curtosis = st.text_input("Enter curtosis:")
entropy = st.text_input("Enter entropy:")

try:
    # Convert inputs to float
    data = np.array([[float(variance), float(skewness), float(curtosis), float(entropy)]])
except ValueError:
    st.write("Please enter valid numeric values.")

# Prediction
if st.button("Predict") and 'data' in locals():
    predictionsGaussianNB = modelGaussianNB.predict(data)
    predictionsKNeighborsClassifier = modelKNeighborsClassifier.predict(data)
    predictionsPerceptron = modelPerceptron.predict(data)
    predictionsSVM = modelSVM.predict(data)

    # Display predictions
    st.write("Prediction of GaussianNB:", predictionsGaussianNB[0])
    st.write("Prediction of KNeighborsClassifier:", predictionsKNeighborsClassifier[0])
    st.write("Prediction of Perceptron:", predictionsPerceptron[0])
    st.write("Prediction of SVM:", predictionsSVM[0])
