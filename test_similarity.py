import json
import numpy as np
import faiss
from FlagEmbedding import BGEM3FlagModel

def normalize(embs):
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.clip(norms, 1e-12, None)

with open('chunks.json','r',encoding='utf-8') as f:
    chunks = json.load(f)

idx = faiss.read_index('faiss.index')
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

q = 'điện thoại samsung bảo hành bao lâu'
emb = model.encode([q], max_length=256)['dense_vecs']
emb = normalize(np.array(emb).astype('float32'))

D, I = idx.search(emb, k=len(chunks))

print(f'\nQuery: "{q}"')
print(f'Total chunks: {len(chunks)}')
print('\nTop 10 similarity scores:')
for i in range(min(10, len(D[0]))):
    score = float(D[0][i])
    chunk_idx = int(I[0][i])
    chunk = chunks[chunk_idx]
    print(f'\n{i+1}. Similarity={score:.4f} (threshold=0.7)')
    print(f'   File: {chunk["file"]} | Page: {chunk["page"]}')
    print(f'   Text: {chunk["text"][:200]}...')
    if score >= 0.7:
        print('   ✓ PASS threshold')
    elif score >= 0.6:
        print('   ~ Close to threshold')
    else:
        print('   ✗ Below threshold')
