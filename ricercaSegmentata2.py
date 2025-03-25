import google.generativeai as genai
import numpy as np
from pymilvus import Collection, connections

# Funzione per cercare i segmenti adiacenti e concatenarli
def find_relevant_segment_with_context(query, collection):
    # Crea l'embedding della query
    query_embedding = genai.embed_content(model='models/embedding-001',
                                          content=query,
                                          task_type="RETRIEVAL_QUERY")
    query_vector = query_embedding["embedding"]
    
    # Ricerca per il segmento più simile
    search_results = collection.search([query_vector], "embedding", limit=5, param={"metric_type": "L2"})
    
    # Trova il segmento con la distanza minima
    best_segment = search_results[0][0]  # Ottieni il primo risultato più vicino
    
    # Trova l'indice del segmento trovato
    segment_id = best_segment.id
    
    # Ottieni tutti i segmenti
    all_segments = collection.query(expr="id == {}".format(segment_id), output_fields=["id", "embedding", "text"])
    
    # Aggiungi i segmenti adiacenti per creare un contesto completo
    segment_index = all_segments[0]["id"]
    relevant_segments = []
    
    # Aggiungi il segmento trovato
    relevant_segments.append(all_segments[0]["text"])

    # Cerca segmenti precedenti e successivi
    previous_segment = collection.query(expr="id == {}".format(segment_index - 1), output_fields=["id", "text"])
    next_segment = collection.query(expr="id == {}".format(segment_index + 1), output_fields=["id", "text"])
    
    if previous_segment:
        relevant_segments.insert(0, previous_segment[0]["text"])
    if next_segment:
        relevant_segments.append(next_segment[0]["text"])

    # Concatenare i segmenti
    full_context = "\n".join(relevant_segments)
    
    return full_context

# Esegui la ricerca con la query
query = "VOGLIO CONOSCERE LO STATO DI UNA RDI O EFFETTUARE UN SOLLECITO"
connections.connect(host="localhost", port="19530")
collection = Collection("document_segments")

relevant_passage = find_relevant_segment_with_context(query, collection)
print("Segmento completo trovato:")
print(relevant_passage)
