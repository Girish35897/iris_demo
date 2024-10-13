import streamlit as st
import pickle
import numpy as np

st.title("""
Iris-Classifier
""")

model_path = "model.pkl"
# Loading the model
model = pickle.load(open(model_path,"rb"))

def predict_flower(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm,model):
    #Prepare the query data as required to pass for the model
    query = np.array([SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]).reshape(1,-1)

    prediction = model.predict(query)
    return prediction.item()
# Row 1: Two columns
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(label="sepal_length",value=None,format="%.2f")
with col2:
    sepal_width = st.number_input(label="sepal_width",value=None,format="%.2f")

# Row 2: Two columns
col3, col4 = st.columns(2)
with col3:
    petal_length = st.number_input(label="petal_length",value=None,format="%.2f")
with col4:
    petal_width = st.number_input(label="petal_width",value=None,format="%.2f")


if st.button("Predict"):
  pred = predict_flower(sepal_length,sepal_width,petal_length,petal_width,model)
  st.write("Output: ",pred)