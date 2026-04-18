import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import string

nltk.download("stopwords", quiet=True)


def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))  # Fix: punctuation (not punctuations)
    stop_words = set(stopwords.words("english"))
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)


def match_resume(resume, job_desc):
    resume_clean = preprocess(resume)       # Fix: preprocess (consistent name)
    job_clean = preprocess(job_desc)

    vectorizer = TfidfVectorizer()          # Fix: TfidfVectorizer (correct spelling)
    vectors = vectorizer.fit_transform([resume_clean, job_clean])

    score = cosine_similarity(vectors[0], vectors[1])[0][0]  # Fix: cosine_similarity
    match_percent = round(score * 100, 2)

    job_keywords = set(job_clean.split())
    resume_keywords = set(resume_clean.split())  # Fix: resume_clean (not resume_sclean)
    missing = job_keywords - resume_keywords

    return match_percent, missing


# --- Streamlit UI ---
st.title("📄 Resume Keyword Matcher")         # Fix: st.title() is not used in an if-condition

resume = st.text_area("Paste your Resume here")
job_desc = st.text_area("Paste the Job Description here")  # Fix: proper indentation

if st.button("Check Match"):
    if resume and job_desc:
        score, missing = match_resume(resume, job_desc)
        st.metric("Match Score", f"{score}%")
        st.subheader("🔴 Missing Keywords")
        st.write(", ".join(list(missing)[:20]))
    else:
        st.warning("Please paste both your resume and the job description.")
