import streamlit as st
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------- Load Dataset --------
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/prasannamudde/emotion-Detector/main/emotions.txt'
    df = pd.read_csv(url, names=['text', 'label'], sep=';', engine='python')
    return df

# -------- Clean Text --------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

# -------- Train the Model --------
@st.cache_resource
def train_model(df):
    df = df[df['text'].apply(lambda x: isinstance(x, str))]
    df['clean_text'] = df['text'].apply(clean_text)
    df = df[df['clean_text'].str.strip() != ""]

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, vectorizer, acc

# -------- Streamlit UI --------
def main():
    st.set_page_config(page_title="Mental Health Chatbot", page_icon="🧠")
    st.title("🧠 Mental Health Emotion Detection Chatbot")
    st.markdown("Type in your thoughts and I’ll detect your emotional state.")

    with st.spinner("🔄 Training the model..."):
        df = load_data()
        model, vectorizer, acc = train_model(df)

    st.success(f"✅ Model trained with **{acc * 100:.2f}% accuracy**")

    user_input = st.text_area("💬 How are you feeling today?", height=150)

    if st.button("Analyze Emotion"):
        if user_input.strip() == "":
            st.warning("Please type something.")
        else:
            cleaned = clean_text(user_input)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]
            st.markdown(f"### 🤔 You may be feeling: **{prediction.upper()}**")

    st.markdown("---")
    st.caption("🔒 Your data is private and never stored.")

# -------- Entry Point --------
if __name__ == "__main__":
    main()
