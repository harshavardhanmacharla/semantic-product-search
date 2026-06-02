import pandas as pd
import numpy as np
import os
import re
import difflib
from django.conf import settings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- LOAD DATA ---------------- #

file_path = os.path.join(settings.BASE_DIR, 'data', 'C:\\Users\\Bhargavi\\shop_ai\\data\\Fashion Dataset .csv')
df = pd.read_csv(file_path)

# Keep required columns
df = df[['name', 'price', 'description', 'colour', 'img']]

# Clean data
df['name'] = df['name'].fillna('').str.lower()
df['description'] = df['description'].fillna('').str.lower()
df['colour'] = df['colour'].fillna('').str.lower()

# Combine text
df['combined'] = df['name'] + " " + df['description']

# ---------------- LOAD MODEL + EMBEDDINGS ---------------- #

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = np.load(os.path.join(settings.BASE_DIR, 'data', 'embeddings.npy'))

# ---------------- SPELLING CORRECTION ---------------- #

all_words = set()
for name in df['name']:
    for word in name.split():
        all_words.add(word)

def correct_query(query):
    corrected = []
    for word in query.split():
        matches = difflib.get_close_matches(word, all_words, n=1, cutoff=0.6)
        corrected.append(matches[0] if matches else word)
    return " ".join(corrected)

# ---------------- PRODUCT DETECTION ---------------- #

known_products = [
    'tshirt','shirt','jeans','trousers','pants',
    'hoodie','jacket','sweatshirt','blazer',
    'dress','kurta','top','skirt'
]

def extract_product(query):
    for word in query.split():
        if word in known_products:
            return word
    return None

# ---------------- PRICE FILTER ---------------- #

def extract_price(query):
    match = re.search(r'under (\d+)', query)
    if match:
        return int(match.group(1))
    return None

# ---------------- COLOR FILTER ---------------- #

colors = ['black','white','red','blue','green','yellow','pink','grey']

def extract_color(query):
    for color in colors:
        if color in query:
            return color
    return None

# ---------------- MAIN SEARCH FUNCTION ---------------- #

def smart_search(query):
    query = query.lower()

    # Step 1: spelling correction
    query = correct_query(query)

    # Step 2: extract filters
    product = extract_product(query)
    price_limit = extract_price(query)
    color = extract_color(query)

    filtered_df = df.copy()

    # Step 3: apply filters
    if product:
        filtered_df = filtered_df[filtered_df['name'].str.contains(product)]

    if price_limit:
        filtered_df = filtered_df[filtered_df['price'] <= price_limit]

    if len(filtered_df) == 0:
        filtered_df = df.copy()

    # 🔥 Step 4: PRIORITY BOOSTING
    filtered_df['priority'] = 0

    if product:
        filtered_df.loc[filtered_df['name'].str.contains(product), 'priority'] += 1

    if color:
        filtered_df.loc[filtered_df['colour'].str.contains(color), 'priority'] += 1

    # 🔥 Step 5: embeddings (FAST - using indices)
    filtered_indices = filtered_df.index.tolist()
    filtered_embeddings = embeddings[filtered_indices]

    query_embedding = model.encode([query])
    similarity = cosine_similarity(query_embedding, filtered_embeddings)[0]

    # 🔥 Step 6: combine score
    filtered_df['similarity'] = similarity

    # FINAL SCORE = priority + similarity
    filtered_df['final_score'] = filtered_df['priority'] + filtered_df['similarity']

    # 🔥 Step 7: sort
    filtered_df = filtered_df.sort_values(by='final_score', ascending=False)

    results = filtered_df[['name', 'price', 'img']].head(5)

    return results.to_dict(orient='records')