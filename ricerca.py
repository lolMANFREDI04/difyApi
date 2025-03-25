import numpy as np
import pandas as pd
import textwrap
import google.generativeai as genai

# Configura la chiave API di Google AI
genai.configure(api_key="AIzaSyAatePm0lWyJXSlK7iSnHpy2Zr0ExQCKL0")

# Specifica i modelli per embedding e generazione
embedding_model = 'models/embedding-001'
generation_model_name = 'gemini-1.5-pro-latest'
generation_model = genai.GenerativeModel(generation_model_name)

# Contenuto del documento (estratto, ad es. dalla sezione "1. VOGLIO SEGNALARE UN’ANOMALIA" e parti correlate)
document_text = r"""
1.	VOGLIO SEGNALARE UN’ANOMALIA
Nel momento in cui l'utente crea una RDI aggiungere i record: gruppo di lavoro e la data e ora in cui l'utente è presente in ufficio.
Nel momento in cui si inserisce il codice dell’asset e la Sede è sbagliata seguire il seguente flusso:
1)	Inserisci l’inventario del PLI
2)	Il tuo PLI è ….? Sì\No
-	Sì → continua (Inserisci la stanza, Inserisci il piano)
-	No → contattaci al seguente indirizzo e-mail: contactcenter@sispi.it o protocollo@sispi.it
Nel momento in cui l’utente inserisce il suo numero di cellulare seguire il seguente flusso:
1)	Scrivi il tuo numero di telefono, possibilmente cellulare, per poterti rintracciare
2)	Confermi il numero di cellulare? Sì\No
Sì → continua
No → Inserisci il numero

Aggiungere il PULSANTE “indietro” ogni volta che si seleziona un’opzione

L'ANOMALIA RIGUARDA UN PLI:
	SPECIFICARE CHE L’ASSET VIENE INDENTIFICATO CON LA DICITURA PLI (per computer)
Quando si seleziona l’opzione “L’anomalia riguarda un PLI” cambiare la dicitura “Hai inserito il PLI” in “Hai selezionato PLI”.
[... Altre sezioni relative a problematiche hardware e software ...]
"""

# Creazione del dataframe con una sola riga
# La colonna 'Embeddings' conterrà l'embedding generato per il documento
# Si specifica il task_type "retrieval_document" per i testi di riferimento
document_embedding_response = genai.embed_content(
    model=embedding_model,
    content=document_text,
    task_type="retrieval_document",
    title="Documento di segnalazione anomalie"
)
document_embedding = document_embedding_response["embedding"]

df = pd.DataFrame({
    'Text': [document_text],
    'Embeddings': [document_embedding]  # Assicurati che l'embedding sia una lista di float
})

# Funzione per trovare il passaggio migliore (in questo caso, l'intero documento)
def find_best_passage(query, dataframe):
    """
    Calcola il prodotto scalare tra l'embedding della query e quella di ogni documento nel dataframe.
    Ritorna il testo del passaggio con il valore massimo.
    """
    query_embedding_response = genai.embed_content(
        model=embedding_model,
        content=query,
        task_type="retrieval_query"
    )
    query_embedding = query_embedding_response["embedding"]
    dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding)
    idx = np.argmax(dot_products)
    return dataframe.iloc[idx]['Text']

# Funzione per costruire il prompt per il modello di generazione
def make_prompt(query, relevant_passage):
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = textwrap.dedent("""\
        You are a helpful and informative bot that answers questions using text from the reference passage included below.
        Be sure to respond in a complete sentence, being comprehensive, including all relevant background information.
        However, you are talking to a non-technical audience, so be sure to break down complicated concepts and strike a friendly and conversational tone.
        If the passage is irrelevant to the answer, you may ignore it.
        QUESTION: '{query}'
        PASSAGE: '{relevant_passage}'

          ANSWER:
    """).format(query=query, relevant_passage=escaped)
    return prompt

# Esempio di utilizzo del sistema Q&A con il contenuto del documento
if __name__ == "__main__":
    # Query riguardante il contenuto del documento
    query = "Come posso segnalare un'anomalia quando l'utente crea una RDI?"
    
    # Trova il passaggio più rilevante dal documento
    passage = find_best_passage(query, df)
    print("Passaggio rilevante trovato:")
    print(passage)
    
    # Costruisci il prompt per la generazione della risposta
    prompt = make_prompt(query, passage)
    print("\nPrompt per la generazione:")
    print(prompt)
    
    # Genera la risposta usando il modello Gemini
    answer = generation_model.generate_content(prompt)
    print("\nRisposta generata:")
    print(answer.text)
