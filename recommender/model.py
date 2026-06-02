import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
df = pd.read_csv("Fashion Dataset.csv")
df = df[['name', 'price', 'colour', 'brand', 'description']]

df['name'] = df['name'].fillna('').str.lower()
df['description'] = df['description'].fillna('').str.lower()
df['colour'] = df['colour'].fillna('').str.lower()

df['combined'] = df['name'] + " " + df['description'] 
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined'])
def smart_search(query):
    
    query = query.lower()
    
    is_hoodie = 'hoodie' in query
    is_black = 'black' in query
    
    price_limit = None
    if 'under' in query:
        words = query.split()
        for i, word in enumerate(words):
            if word == 'under' and i+1 < len(words):
                try:
                    price_limit = int(words[i+1])
                except:
                    pass

    filtered = df.copy()
    
    if is_hoodie:
        filtered = filtered[filtered['name'].str.contains('hoodie', na=False)]
    
    if is_black:
        filtered = filtered[filtered['colour'].str.contains('black', na=False)]
    
    if price_limit:
        filtered = filtered[filtered['price'] < price_limit]

    if len(filtered) == 0:
        filtered = df.copy()
        
        if is_hoodie:
            filtered = filtered[filtered['name'].str.contains('hoodie', na=False)]
        
        if is_black:
            filtered = filtered[filtered['colour'].str.contains('black', na=False)]

    if len(filtered) == 0:
        filtered = df[df['name'].str.contains('hoodie', na=False)]

    query_vec = tfidf.transform([query])
    
    filtered_indices = filtered.index
    filtered_matrix = tfidf_matrix[filtered_indices]
    
    similarity = cosine_similarity(query_vec, filtered_matrix)
    
    top_indices = similarity[0].argsort()[-5:][::-1]
    
    results = filtered.iloc[top_indices][['name', 'price']]
    
    return results
