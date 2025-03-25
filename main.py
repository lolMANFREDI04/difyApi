from flask import Flask, request, jsonify
from ricercaSegmentata import initial

app = Flask(__name__)

def query_milvus(query, top_k, score_threshold, metadata_condition=None):
    """
    Simula la ricerca su Milvus.
    In un'implementazione reale, qui andrai a utilizzare il client Milvus per eseguire una ricerca vettoriale.
    """

    # results = initial(query)

    if(query=="VOGLIO SEGNALARE UN’ANOMALIA"):
        results= r"""
    1.	VOGLIO SEGNALARE UN’ANOMALIA
Nel momento in cui l'utente crea una RDI aggiungere i record: gruppo di lavoro e la data e ora in cui l'utente è presente in ufficio.
Nel momento in cui si inserisce il codice dell’asset e la Sede è sbagliata seguire il seguente flusso:
1)	Inserisci l’inventario del PLI
2)	Il tuo PLI è ….? Sì\No
-	Sì  continua (Inserisci la stanza, Inserisci il piano)
-	No  contattaci al seguente indirizzo e-mail: contactcenter@sispi.it o protocollo@sispi.it
Nel momento il cui l’utente inserisce il suo numero di cellulare seguire il seguente flusso:
1)	Scrivi il tuo numero di telefono, possibilmente cellulare, per poterti rintracciare
2)	Confermi il numero di cellulare? Sì\No
Sì  continua
No  Inserisci il numero

 	Aggiungere il PULSANTE “indietro” ogni volta che si seleziona un’opzione"""
        
    else:

        results= r"""
            L'ANOMALIA RIGUARDA UN PLI:
    	SPECIFICARE CHE L’ASSET VIENE INDENTIFICATO CON LA DICITURA PLI (per computer)
    Quando si seleziona l’opzione “L’anomalia riguarda un PLI” cambiare la dicitura “Hai inserito il PLI” in “Hai selezionato PLI”.
        PROBLEMATICA HARDWARE:
    •	Aggiungere l’opzione “il monitor non si accende?”
    -	Verifica se il cavo è collegato correttamente. Adesso il monitor si è acceso? Sì\No
    Si  Ok
    No  Apri RDI (assegnare al tecnico di zona)
    •	Il pc non si accende? 
    Verifica che il cavo di alimentazione sia collegato in maniera corretta. Se il tasto di alimentazione dietro il PC si trova in posizione "O" è necessario portarlo in posizione "I" e provare nuovamente ad accendere il PC. Il PC adesso si è acceso? Sì\No
    Sì  Ok
    No  Apri RDI al tecnico di zona 
    •	Il pc è lento e si blocca durante le attività lavorative? 
    Apri RDI al primo livello
    •	Il pc si blocca all’avvio? 
    -	Schermata blu con scritta bianca: crash di sistema operativo (assegnare al tecnico di zona);
    -	Digitare F1: batteria tampone guasta (assegnare al tecnico di zona);
    -	Disk boot failure insert system disk and press enter (assegnare al tecnico di zona);
    -	Schermata nera, messaggio no signal, led del monitor di colore arancione, probabile scheda video guasta (assegnare al tecnico di zona);
    -	Schermata nera senza messaggio. Chiedere all’utente di verificare la luce Led dell’unità centrale vicino al tasto di accensione, se è gialla fissa, probabile scheda madre guasta (assegnare alla logistica per rientro in laboratorio);
    -	Schermata nera, nessun led acceso del monitor, chiedere all’utente, se può verificare la funzionalità del cavo di alimentazione del monitor (per esempio, utilizzando quello della stampante), in caso negativo assegnare alla logistica per la sostituzione del monitor.

    •	Altro spazio note (assegnare al tecnico di zona)"""

    


    # Esempio di risultati dummy
    dummy_results = [
        {
            "content": results,  #.entity.get("text")
            "score": 0.98,
            "title": "knowledge.txt",
            "metadata": {
                "path": "s3://dify/knowledge.txt",
                "description": "dify knowledge document"
            }
         }#,
        # {
        #     "content": "The Innovation Engine for GenAI Applications",
        #     "score": 0.66,
        #     "title": "introduce.txt",
        #     "metadata": {
        #         "path": "s3://dify/introduce.txt",
        #         "description": "dify introduce"
        #     }
        # }
    ]
    # Filtra i risultati in base allo score_threshold e ritorna i primi top_k risultati
    filtered_results = [doc for doc in dummy_results if doc["score"] >= score_threshold]
    return filtered_results[:top_k]

@app.route('/retrieval', methods=['POST'])
def retrieval():
    # Verifica dell'header di autorizzazione
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith("Bearer "):
        return jsonify({
            "error_code": 1001,
            "error_msg": "Invalid Authorizationcd  header format. Expected 'Bearer <token>' format."
        }), 403

    # Parsing del corpo della richiesta in formato JSON
    data = request.get_json()
    if not data:
        return jsonify({
            "error_code": 400,
            "error_msg": "Bad Request: JSON data is missing"
        }), 400

    knowledge_id = data.get("knowledge_id")
    query_text = data.get("query")
    retrieval_setting = data.get("retrieval_setting")
    if not knowledge_id or not query_text or not retrieval_setting:
        return jsonify({
            "error_code": 400,
            "error_msg": "Missing required fields: knowledge_id, query and retrieval_setting are mandatory."
        }), 400

    # Estrazione dei parametri per la ricerca
    top_k = retrieval_setting.get("top_k", 5)
    score_threshold = retrieval_setting.get("score_threshold", 0.5)
    metadata_condition = data.get("metadata_condition", None)

    # Esecuzione della ricerca su Milvus (funzione simulata)
    results = query_milvus(query_text, top_k, score_threshold, metadata_condition)

    # Restituzione del risultato in formato JSON conforme alle specifiche
    response = {
        "records": results
    }
    return jsonify(response), 200

if __name__ == '__main__':
    app.run(debug=True)