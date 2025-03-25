from pymilvus import connections, Collection

# Connessione al server Milvus (assicurati che host e port siano corretti)
connections.connect(alias="default", host="localhost", port="19530")

# Ora puoi accedere alla collezione
collection = Collection("document_embeddings")
print(f"Numero di entità nella collezione: {collection.num_entities}")


import numpy as np

# Recupera la collezione
collection = Collection("document_embeddings")

# Definisci una query embedding (usa un array casuale se non ne hai una vera)
query_embedding = np.random.rand(1, 768).astype("float32")  # Usa la stessa dimensione della tua embedding

# Esegui la ricerca
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
results = collection.search(query_embedding, "embedding", search_params, limit=1)

# Stampa i risultati
for hit in results:
    print(f"ID: {hit[0].id}, Distance: {hit[0].distance}")
