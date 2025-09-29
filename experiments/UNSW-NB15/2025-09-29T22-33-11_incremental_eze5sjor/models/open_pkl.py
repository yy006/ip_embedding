from gensim.models import Word2Vec

path = "model_block_001"  # 例: pklでもOK
model = Word2Vec.load(path)

print("model:", model.wv['192.168.10.3'])
print("vocab size:", len(model.wv.index_to_key))
print("vector size:", model.wv.vector_size)
print("top10:", model.wv.index_to_key[:10])

# ベクトル取得
vec = model.wv.get_vector(model.wv.index_to_key[0])
print("vec shape:", vec.shape)