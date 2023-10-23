import streamlit as st
from gensim.models import Word2Vec

# モデルの読み込み（事前にトレーニングして保存しておく必要があります）
model = Word2Vec.load("word2vec_example.model")

def get_similar_words(query_word):
    if query_word not in model.wv:
        return [(f"'{query_word}' is not in the vocabulary.", 0.0)]
    else:
        return model.wv.most_similar(query_word, topn=1)

st.title("ダークライ構文")

query_word = st.text_input("その単語ダークライ構文にします。:", )

similar_words = get_similar_words(query_word)
st.write(f"{query_word} vs {similar_words} vs ダークライ")

# for word, score in similar_words:
#     st.write(f"{word}: {score:.4f}")
