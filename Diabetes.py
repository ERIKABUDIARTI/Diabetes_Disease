#Prepare Libraries
import streamlit as st
import pandas as pd
import pickle
import time
from PIL import Image

#Setting page
st.set_page_config(page_title="Modelling Page",
                   layout="wide")

#Introduction
st.write("""
         # Welcome to [ERIKA](https://www.linkedin.com/in/erika-budiarti/)'s Machine Learning Dashboard
        """)

st.write("""
        # This app predicts the **Diabetes Disease**.
        
        Data obtained from the [Diabetes Disease dataset](https://www.kaggle.com/datasets/saurabh00007/diabetescsv)
        """)

#Collects user input features into dataframe
st.sidebar.header('User Input Features:')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else: 
    def user_input_features():
        st.sidebar.header('Manual Input')
        Glucose = st.sidebar.slider('Glucose Level : Plasma glucose concentration a 2 hours in an oral glucose tolerance test', 40, 200, 40, step = 20)
        BMI = st.sidebar.slider('Body Mass Index (BMI) : weight in kg/(height in m)^2', 15, 50, 15, step=5)
        Age = st.sidebar.slider('Age of the Person(year)', 20, 70, 20, step=5)
        Pregnancies = st.sidebar.slider('Number of Pregnancies', 0, 15, 0, step=3)
        DiabetesPedigreeFunction = st.sidebar.slider('Diabetes Pedigree Index : Genetic risk value', 0.5, 2.0, 0.5, step=0.5)
        Insulin = st.sidebar.slider('Insulin Level : 2-Hour serum insulin(mu U/ml)', 0, 350, 0, step=50)

        data = {'Glucose' : Glucose,
                'BMI' : BMI,
                'Age' : Age,
                'Pregnancies': Pregnancies,
                'DiabetesPedigreeFunction' : DiabetesPedigreeFunction,
                'Insulin' : Insulin}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

loaded_model = None
diagnosis = " "

# Add picture  
img = Image.open("diabetes-ilust.jpg")
st.image(img, width=500)

#Loading images
happy = Image.open('happy-emoji.png')
sad = Image.open('crying-emoji.png')

if st.sidebar.button('Diagnosis!'):
    df = input_df
    st.write(df)
    with open("best_model_nb.pkl", 'rb') as file:  
        loaded_model = pickle.load(file)

if loaded_model is not None:        
        prediction = loaded_model.predict(df) 
        if (prediction == 0):
            result = 'You Are Healthy'
        else:
            result = 'You Have Diabetes Disease'

        diagnosis = str(result)
        
        st.subheader('Prediction Result: ')        
     
        with st.spinner('Wait for it...'): 
                time.sleep(4)
        st.success(f"Prediction of this app is {diagnosis}")  
 
        if (diagnosis == 'You Are Healthy'):
            st.image(happy)
        else:
            st.image(sad)
        

    
        
    
