import joblib
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
clf = joblib.load('spam_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

def prediction(vector_content):
    predict = clf.predict(vector_content)
    return predict

def main():
    st.title('Email Spam Detector')
    text = st.text_input('Enter text')
    vector_content = vectorizer.transform(text.split())
    if st.button('Predict'):
        result = prediction(vector_content)
        if(result[0]==1):
            st.success('Email is Spam!!')
        else:
            st.success('Email is Safe!!')


if __name__=='__main__':
    main()