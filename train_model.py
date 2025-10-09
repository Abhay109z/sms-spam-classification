import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import pickle

# 1. Read dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]  # keep only labels and messages
df = df.rename(columns={'v1': 'label', 'v2': 'message'})  # rename

# 2. Convert labels to numbers
# ham = 0, spam = 1
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])

# 3. Convert text to numbers (TF-IDF)
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['message'])
y = df['label']

# 4. Train model
model = MultinomialNB()
model.fit(X, y)

# 5. Save model and vectorizer
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(model, open('model.pkl', 'wb'))

print("✅ Model trained and saved!")
