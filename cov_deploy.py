import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Early Covid-19 Detection App
This app predicts the **Covid-19** symptomps!
""")

st.sidebar.header('User Input Features')

# st.sidebar.markdown("""
# [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
# """)

# Collects user input features into dataframe
def user_input_features():
        Breating_problem = st.sidebar.selectbox('Breathing Problem',('Yes','No'))
        Fever = st.sidebar.selectbox('Fever',('Yes','No'))
        Dry_Cough = st.sidebar.selectbox('Dry Cough',('Yes','No'))
        Sore_throat = st.sidebar.selectbox('Sore throat',('Yes','No'))
        Running_Nose = st.sidebar.selectbox('Running Nose',('Yes','No'))
        Asthma = st.sidebar.selectbox('Asthma',('Yes','No'))
        Chronic_Lung_Disease = st.sidebar.selectbox('Chronic Lung Disease',('Yes','No'))
        Headache = st.sidebar.selectbox('Headcache',('Yes','No'))
        Heart_Disease= st.sidebar.selectbox('Heart Disease',('Yes','No'))
        Diabetes = st.sidebar.selectbox('Diabetes',('Yes','No'))
        HyperTension = st.sidebar.selectbox('Hyper Tension',('Yes','No'))
        Fatigue  = st.sidebar.selectbox('Fatigue',('Yes','No'))
        Gastrointestinal = st.sidebar.selectbox('Gastrointestinal',('Yes','No'))
        Abroad_travel = st.sidebar.selectbox('Abroad travel',('Yes','No'))
        Contact = st.sidebar.selectbox('Contact with COVID Patient',('Yes','No'))
        Attended = st.sidebar.selectbox('Attended Large Gathering',('Yes','No'))
        Visited = st.sidebar.selectbox('Visited Public Exposed Places',('Yes','No'))
        Family = st.sidebar.selectbox('Family working in Public Exposed Places',('Yes','No'))
        Masks = st.sidebar.selectbox('Wearing Masks',('Yes','No'))
        Sanitizing = st.sidebar.selectbox('Sanitization from Market',('Yes','No'))
        data = {'Breathing Problem' : Breating_problem,
                'Fever' : Fever,
                'Dry Cough' : Dry_Cough,
                'Sore throat' : Sore_throat,
                'Running Nose' : Running_Nose ,
                'Asthma' : Asthma,
                'Chronic Lung Disease' : Chronic_Lung_Disease,
                'Headache' : Headache,
                'Heart Disease' : Heart_Disease,
                'Diabetes' : Diabetes,
                'Hyper Tension' : HyperTension,
                'Fatigue' : Fatigue,
                'Gastrointestinal' : Gastrointestinal,
                'Abroad travel' : Abroad_travel,
                'Contact with COVID Patient' : Contact,
                'Attended Large Gathering' : Attended,
                'Visited Public Exposed Places' : Visited,
                'Family working in Public Exposed Places' : Family,
                'Wearing Masks' : Masks,
                'Sanitization from Market' : Sanitizing}
        features = pd.DataFrame(data, index=[0])
        features = np.where(features.values == 'Yes',1,0)
        features = pd.DataFrame(features, index=[0])
        return features
input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
# penguins_raw = pd.read_csv('penguins_cleaned.csv')
# penguins = penguins_raw.drop(columns=['species'])
# df = pd.concat([input_df,penguins],axis=0)

# # Encoding of ordinal features
# # https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
# encode = ['sex','island']
# for col in encode:
#     dummy = pd.get_dummies(df[col], prefix=col)
#     df = pd.concat([df,dummy], axis=1)
#     del df[col]
# df = df[:1] # Selects only the first row (the user input data)

# # Displays the user input features
# st.subheader('User Input features')

# if uploaded_file is not None:
#     st.write(df)
# else:
#     st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
#     st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('covid_DT.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(input_df)

# st.subheader('Prediction')
if(prediction == 0) :
    st.write("""
    # COVID DETECTION = NEGATIVE
    Congratulations!
    """)
else:
    st.write("""
    # COVID DETECTION = POSITIVE
    Go get a COVID-19 test immediately!
    """)

st.write("""
## About the Model
""")

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score

X_test = pickle.load(open('covid_Xtest.pkl', 'rb'))
y_test = pickle.load(open('covid_Ytest.pkl', 'rb'))
Y_pred = load_clf.predict(X_test)

cm = confusion_matrix(y_test, Y_pred)
fig, ax = plt.subplots()
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
st.write(fig)

accuracy = accuracy_score(y_test,Y_pred)
precision = precision_score(y_test, Y_pred,average='micro')
recall = recall_score(y_test, Y_pred,average='micro')
f1 = f1_score(y_test,Y_pred,average='micro')
st.write("Model Accuracy", accuracy)
st.write("Model Precision", precision)
st.write("Model Recall", recall)
st.write("Model F1-Score", f1)
