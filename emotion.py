import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------- Step 1: Load and Clean Data --------
file_path = 'C:/emotion detector/emotions.txt'
df = pd.read_csv(file_path, names=['text', 'label'], sep=';', engine='python')
 # common issue fix: set correct separator


# Clean text function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

# Drop non-string rows
df = df[df['text'].apply(lambda x: isinstance(x, str))]

# Apply cleaning
df['clean_text'] = df['text'].apply(clean_text)
df = df[df['clean_text'].str.strip() != ""]

# -------- Step 2: Vectorize and Train Model --------
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
from sklearn.utils import class_weight
model = LogisticRegression(class_weight='balanced', max_iter=1000)  # ensure convergence
model.fit(X_train, y_train)

# -------- Step 3: Chat Interface --------
print("\n🤖 Mental Health Chatbot")
print("Type your thoughts below. Type 'exit' to quit.\n")

while True:
    try:
        user_input = input("🗨 You: ")
        if user_input.lower() == 'exit':
            print("👋 Goodbye! Stay healthy.")
            break

        cleaned_input = clean_text(user_input)
        if cleaned_input.strip() == "":
            print("🤖 Bot: Please enter more descriptive thoughts.\n")
            continue

        vector_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vector_input)[0]
        print(f"🤖 Bot: You may be feeling *{prediction.upper()}*.\n")
    except Exception as e:
        print(f"❌ Error: {e}")
