import streamlit as st
from transformers import pipeline

summarizer = pipeline("summarization", model="ayush0205/medicalSummarizer")

import streamlit as st

st.title("⚕️✨  Medical Summarizer")



def generate_response(input_text):
    st.info(summarizer("summarize: "+ input_text)[0]['summary_text'])


with st.form("my_form"):
    text = st.text_area(
        "Enter long patient history:",
        "Sample Text",
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(text)
