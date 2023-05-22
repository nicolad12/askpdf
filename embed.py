import streamlit as st
import pandas as pd
import numpy as np
import tiktoken
import openai
from openai.embeddings_utils import get_embedding

embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

# import pdf as a dataframe of data split in pages

# feature extraction through embeddings

# save embeddings in a file

input_datapath1 = "./reviews_lim.csv"

@st.cache_data
df = pd.load_data(input_datapath1, index_col=0)
df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]
df = df.dropna()
df["combined"] = (
    "Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
)

st.dataframe(df.head(2))

# subsample to 1k most recent reviews and remove samples that are too long
top_n = 1000
df = df.sort_values("Time").tail(top_n * 2)  # first cut to first 2k entries, assuming less than half will be filtered out
df.drop("Time", axis=1, inplace=True)

encoding = tiktoken.get_encoding(embedding_encoding)

# omit reviews that are too long to embed
df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))
df = df[df.n_tokens <= max_tokens].tail(top_n)

# use tiktoken library to count tokens in a string

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

num_tokens_from_string("tiktoken is great!", "cl100k_base")

# Ensure you have your API key set in your environment per the README: https://github.com/openai/openai-python#usage

openai.api_key ='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'

# This may take a few minutes !!! This is the most costly task
@st.cache_data
df["embedding"] = df.combined.apply(lambda x: get_embedding(x, engine=embedding_model))
df.to_csv("./fine_food_reviews_with_embeddings_1k.csv")

@st.cache_data
input_datapath2 ="./fine_food_reviews_with_embeddings_1k.csv"
df = pd.read_csv(input_datapath2)
df["embedding"] = df.embedding.apply(eval).apply(np.array)

# compare the cosine similarity of the embeddings of the query and the documents, and show top_n best matches.

from openai.embeddings_utils import get_embedding, cosine_similarity

# search through the reviews for a specific product
def search_reviews(df, product_description, n=3, pprint=True):
    product_embedding = get_embedding(
        product_description,
        engine="text-embedding-ada-002"
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .combined.str.replace("Title: ", "")
        .str.replace("; Content:", ": ")
    )
    if pprint:
        for r in results:
            print(r[:200])
            print()
    return results

txt = st.text_area('Text to analyze', '''
    jamaica beans
    ''')
results = search_reviews(df, txt , n=3)
st.write("Text to search: 'jamaica beans'")
st.write(results)
