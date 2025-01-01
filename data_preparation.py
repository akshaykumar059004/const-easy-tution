import json
import pandas as pd

# Load dataset
with open('legal_data.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Ensure 'Question' and 'Answer' columns exist
if 'Question' not in df.columns or 'Answer' not in df.columns:
    raise ValueError("JSON file must contain 'Question' and 'Answer' fields")

# Preprocess data
def preprocess_text(text):
    # Add any preprocessing steps here (e.g., lowercasing, removing special characters)
    return text

df['Question'] = df['Question'].apply(preprocess_text)
df['Answer'] = df['Answer'].apply(preprocess_text)

# Shuffle rows
df = df.sample(frac=1).reset_index(drop=True)

# Save preprocessed and shuffled data
df.to_csv('preprocessed_legal_data1.csv', index=False)
