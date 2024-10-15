
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"./diabetes.csv")

st.title('Diabetes Checkup')
st.header('Patient Data')

x = df.drop(['Outcome'], axis=1)
x = df.drop(['Outcome'], axis=1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

def calc():
    col1, col2, col3 = st.columns(3)
    with col1:
        pregnancies = st.number_input('Pregnancies', min_value=0, max_value=17, value=3)
        bp = st.number_input('Blood Pressure', min_value=0, max_value=122, value=70)
        bmi = st.number_input('BMI', min_value=0, max_value=67, value=20)

    with col2:
        glucose = st.number_input('Glucose', min_value=0, max_value=200, value=120)
        skinthickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
        dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.4, value=0.47)

    with col3:
        insulin = st.number_input('Insulin', min_value=0, max_value=846, value=79)
        age = st.number_input('Age', min_value=21, max_value=88, value=33)

    output = {
        'pregnancies': pregnancies,
        'glucose': glucose,
        'bp': bp,
        'skinthickness': skinthickness,
        'insulin': insulin,
        'bmi': bmi,
        'dpf': dpf,
        'age': age
    }
    report_data = pd.DataFrame(output, index=[0])
    return report_data

user_data = calc()
st.subheader('Patient Data')
st.write(user_data)

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
result = rf.predict(user_data)

st.subheader('Your Report: ')
output = 'You are not Diabetic' if result[0] == 0 else 'You are Diabetic'
st.title(output)

st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, rf.predict(x_test)) * 100) + '%')
