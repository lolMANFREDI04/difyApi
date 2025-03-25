import numpy as np
import google.generativeai as genai
from docx import Document
from pymilvus import connections, Collection, DataType, FieldSchema, CollectionSchema, utility

# Configura la chiave API di Google AI
genai.configure(api_key="AIzaSyAatePm0lWyJXSlK7iSnHpy2Zr0ExQCKL0")

def extract_paragraphs(file_path):
    """
    Estrae i paragrafi non vuoti dal documento DOCX.
    """
    doc = Document(file_path)
    # Rimuove paragrafi vuoti e spazi indesiderati
    paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip() != ""]
    return paragraphs

def generate_embedding(text, title):
    """
    Genera l'embedding del testo usando il modello Gemini (task_type: RETRIEVAL_DOCUMENT).
    """
    model = 'models/embedding-001'
    response = genai.embed_content(
        model=model,
        content=text,
        task_type="RETRIEVAL_DOCUMENT",
        title=title
    )
    return response["embedding"]

def connect_to_milvus():
    """
    Connette a Milvus.
    """
    connections.connect(host="localhost", port="19530")

def create_collection():
    """
    Crea (o carica, se già esistente) una collezione in Milvus per i segmenti del documento.
    La collezione include:
      - id: identificatore univoco (auto_id abilitato)
      - embedding: vettore (FLOAT_VECTOR)
      - text: il contenuto del segmento (VARCHAR)
    """
    collection_name = "document_segments"
    
    # Definisce gli schemi dei campi
    field_id = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
    field_embedding = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
    field_text = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000)
    
    schema = CollectionSchema(
        fields=[field_id, field_embedding, field_text],
        description="Segmented Document Embeddings"
    )
    
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
    else:
        collection = Collection(name=collection_name, schema=schema)
        index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
        collection.create_index(field_name="embedding", index_params=index_params)
    
    return collection

def insert_segments(collection, segments, title="Segment"):
    """
    Per ogni segmento, genera l'embedding e poi inserisce il vettore insieme al testo.
    """
    embeddings = []
    texts = []
    
    for segment in segments:
        emb = generate_embedding(segment, title)
        embeddings.append(emb)
        texts.append(segment)
    
    # Inserisce gli enti; il campo "id" è auto-generato
    entities = [
        embeddings,  # campo "embedding"
        texts        # campo "text"
    ]
    
    collection.insert(entities)
    collection.load()

if __name__ == "__main__":
    # Specifica il percorso del file DOCX
    file_path = "document_search.docx"
    
    # Estrae i paragrafi dal documento
    segments = extract_paragraphs(file_path)
    print(f"Segmenti trovati: {len(segments)}")
    
    # Connessione a Milvus e creazione della collezione
    connect_to_milvus()
    collection = create_collection()
    
    # Inserisce i segmenti nella collezione
    insert_segments(collection, segments)
    
    print("Segmenti e relativi embedding inseriti con successo in Milvus!")
