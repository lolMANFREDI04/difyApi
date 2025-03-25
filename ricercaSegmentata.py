import numpy as np
import google.generativeai as genai
from pymilvus import Collection, connections

# Configura la chiave API di Google AI
genai.configure(api_key="AIzaSyAatePm0lWyJXSlK7iSnHpy2Zr0ExQCKL0")
embedding_model = 'models/embedding-001'

def generate_query_embedding(query):
    """
    Genera l'embedding per la query usando il task_type "retrieval_query".
    """
    response = genai.embed_content(
        model=embedding_model,
        content=query,
        task_type="retrieval_query"
    )
    return response["embedding"]

def search_segment(query, collection, top_k=1):
    """
    Cerca nel database vettoriale il segmento più rilevante per la query.
    """
    query_embedding = generate_query_embedding(query)
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
         data=[query_embedding],
         anns_field="embedding",
         param=search_params,
         limit=top_k,
         expr=None,
         output_fields=["text", "id"]   # Aggiunto il campo "id" per recuperare l'indice del segmento
    )
    return results

def get_paragraphs_for_section(segment, collection):
    """
    Recupera tutti i paragrafi associati al titolo, evitando titoli secondari.
    """
    # Estrai il titolo (assumiamo che il titolo sia un numero seguito da punto, es: '2.')
    title = segment['text'].split("\n")[0]  # Prendi la prima riga come titolo
    paragraphs = []

    # Cerca tutti i paragrafi che seguono il titolo, ma fermati quando trovi un nuovo titolo
    is_paragraph = False
    for hit in collection.query(expr="id > {}".format(segment['id']), output_fields=["text"]):
        text = hit["text"]
        if text.strip() == "" or text.startswith("•") or text.startswith("  "):  # Paragrafi (no titoli)
            paragraphs.append(text)
        else:
            # Se trovi un altro titolo (esempio: "1. Titolo"), fermati
            if any(char.isdigit() for char in text.split(" ")[0]):  # Titolo (ad esempio "1.")
                break

    return "\n".join(paragraphs)

def initial(query):
    # Connessione a Milvus e caricamento della collezione
    connections.connect(host="localhost", port="19530")
    collection = Collection("document_segments")
    
    # Query di ricerca
    #query = "2. VOGLIO CONOSCERE LO STATO DI UNA RDI O EFFETTUARE UN SOLLECITO"
    
    # Esegui la ricerca
    results = search_segment(query, collection, top_k=1)
    
    # Stampa il testo del segmento più rilevante
    print("Segmento trovato:")
    for hits in results:
        for hit in hits:
            print("Segmento:", hit.entity.get("text"))
            print("Distanza:", hit.distance)

            # Recupera i paragrafi associati
            paragraphs = get_paragraphs_for_section(hit.entity, collection)
            print("Paragrafi associati al titolo:")
            print(paragraphs)

    return results
