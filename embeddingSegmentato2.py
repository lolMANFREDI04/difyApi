from docx import Document


def extract_paragraphs(file_path):
    """
    Estrae i paragrafi non vuoti dal documento DOCX, raggruppando i titoli con i loro paragrafi correlati.
    Gestisce anche casi speciali in cui due titoli consecutivi devono essere aggregati.
    """
    doc = Document(file_path)
    segments = []
    current_segment = []
    is_title = False
    previous_title = None

    for para in doc.paragraphs:
        text = para.text.strip()
        if text == "":
            continue  # Salta i paragrafi vuoti

        # Identifica i titoli basati su caratteristiche (es. numerazione, maiuscolo, frasi chiave)
        is_current_title = (
            text.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")) or
            text.isupper() or
            text.startswith("L'ANOMALIA RIGUARDA") or
            text.startswith("PROBLEMATICA")
        )

        # Se abbiamo già un segmento in corso e troviamo un nuovo titolo
        if is_current_title and current_segment:
            # Caso speciale: se il titolo precedente è "L'ANOMALIA RIGUARDA..." e il nuovo titolo è "PROBLEMATICA..."
            if previous_title and previous_title.startswith("L'ANOMALIA RIGUARDA") and text.startswith("PROBLEMATICA"):
                # Aggiungi il nuovo titolo al segmento corrente invece di iniziare uno nuovo
                current_segment.append(text)
                is_title = True
                continue

            # Altrimenti, salva il segmento corrente e inizia uno nuovo
            segments.append("\n".join(current_segment))
            current_segment = []

        # Aggiungi il testo al segmento corrente
        if is_current_title:
            current_segment.append(text)
            previous_title = text  # Memorizza il titolo corrente per il controllo successivo
        elif current_segment:
            current_segment.append(text)

        # Resetta lo stato del titolo dopo averlo aggiunto
        is_title = False

    # Aggiungi l'ultimo segmento, se presente
    if current_segment:
        segments.append("\n".join(current_segment))

    return segments