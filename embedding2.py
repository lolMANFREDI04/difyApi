import google.generativeai as genai
from docx import Document
from pymilvus import connections, Collection, DataType, FieldSchema, CollectionSchema

# Imposta la chiave API di Google AI
genai.configure(api_key="AIzaSyAatePm0lWyJXSlK7iSnHpy2Zr0ExQCKL0")

# Leggi il contenuto del file DOCX
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

# Genera l'embedding con Gemini API
def generate_embedding(text, title):
    model = 'models/embedding-001'
    response = genai.embed_content(
        model=model,
        content=text,
        task_type="RETRIEVAL_DOCUMENT",
        title=title
    )
    return response["embedding"]

# Configura la connessione a Milvus
def connect_to_milvus():
    connections.connect(host="localhost", port="19530")

# Crea una collezione in Milvus
def create_collection():
    field_id = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True)
    field_embedding = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
    
    schema = CollectionSchema(fields=[field_id, field_embedding], description="Document Embeddings")
    collection = Collection(name="document_embeddings", schema=schema)
    collection.create_index(field_name="embedding", index_params={"metric_type": "L2"})
    return collection

# Inserisci i dati in Milvus
def insert_to_milvus(collection, embedding):
    collection.insert([[1], [embedding]])  # Inserisce l'embedding con un ID fittizio (1)
    collection.load()

if __name__ == "__main__":
    # Estrai il testo dal file DOCX
    docx_text = extract_text_from_docx("document_search.docx")
    
    # Genera l'embedding
    title = "Documento di ricerca"
    embedding = generate_embedding(docx_text, title)
    
    # Connessione a Milvus e inserimento dati
    connect_to_milvus()
    collection = create_collection()
    insert_to_milvus(collection, embedding)

    print("Embedding inserito con successo in Milvus!")
