import os
import re
import whisper
import stanza
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import yt_dlp

# 1. Download audio from YouTube
def download_audio_ytdlp(youtube_url, output_file="audio.mp3"):
    output_temp = "temp_audio.%(ext)s"
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_temp,
        'quiet': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    if os.path.exists("temp_audio.mp3"):
        os.rename("temp_audio.mp3", output_file)

    return output_file

# 2. Transcribe in Hindi (no translation)
def transcribe_audio(file_path):
    model = whisper.load_model("medium")
    result = model.transcribe(file_path, task="transcribe")
    return result["text"]

# 3. Clean text
def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

# 4. NLP using Stanza for Hindi
def nlp_analysis_hindi(text):
    nlp = stanza.Pipeline(lang='hi', processors='tokenize,pos')
    doc = nlp(text)

    print("\n--- POS Tags ---")
    for sentence in doc.sentences:
        for word in sentence.words:
            print(f"{word.text} -> {word.upos}")

# 5. NER using IndicNER
def nlp_analysis_ner(text):
    print("\n--- Named Entities (IndicNER) ---")
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicNER")
    model = AutoModelForTokenClassification.from_pretrained("ai4bharat/IndicNER")
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    # Break into sentences for better recognition
    sentences = re.split(r'[।.!?]', text)
    entity_found = False

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        entities = ner_pipeline(sentence)
        for ent in entities:
            print(f"{ent['word']} -> {ent['entity_group']}")
            entity_found = True

    if not entity_found:
        print("⚠️ No named entities found.")

# 6. KWIC search
def kwic_search(text, keyword, window=5):
    tokens = text.split()
    for i, token in enumerate(tokens):
        if token == keyword:
            left = ' '.join(tokens[max(i-window, 0):i])
            right = ' '.join(tokens[i+1:i+1+window])
            print(f"...{left} **{token}** {right}...")

# 7. Question Answering
def question_answering(context, question):
    qa = pipeline("question-answering", model="distilbert-base-multilingual-cased", tokenizer="distilbert-base-multilingual-cased")
    result = qa(question=question, context=context)
    print("\n--- QA Result ---")
    print("Q:", question)
    print("A:", result['answer'])

# MAIN
if __name__ == "__main__":
    youtube_link = input("🔗 Enter Hindi YouTube video link: ")

    print("\n🎧 Downloading audio...")
    audio_file = download_audio_ytdlp(youtube_link, output_file="audio.mp3")

    print("📝 Transcribing in Hindi (no translation)...")
    raw_text = transcribe_audio(audio_file)
    transcript = clean_text(raw_text)

    print("\n📄 Transcript Snippet:")
    print(transcript[:500], "...")

    print("\n🔍 NLP Analysis with Stanza (Hindi)...")
    nlp_analysis_hindi(transcript)

    print("\n🔠 NER with IndicNER:")
    nlp_analysis_ner(transcript)



    os.remove(audio_file)
