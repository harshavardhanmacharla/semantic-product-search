import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# load data
df = pd.read_csv("C:\\Users\\Bhargavi\\shop_ai\\data\\Fashion Dataset .csv")

df['name'] = df['name'].fillna('').str.lower()
df['description'] = df['description'].fillna('').str.lower()

df['combined'] = df['name'] + " " + df['description']

# load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# generate embeddings
embeddings = model.encode(df['combined'].tolist())

# save embeddings
np.save("data/embeddings.npy", embeddings)

print(" embeddings saved")