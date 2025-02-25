import streamlit as st
import json
from openai import OpenAI
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
import nltk
import tempfile
import os
from dotenv import load_dotenv

nltk.download("vader_lexicon")
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
sia = SentimentIntensityAnalyzer()

# Streamlit UI
st.title("🎵 Audio-Based Sentiment Analysis")
st.write("Upload an audio file to analyze its sentiment progression over time.")

# Upload audio file
uploaded_file = st.file_uploader("Upload an MP3 or WAV file", type=["mp3", "wav"])

if uploaded_file:
    st.write("✅ File uploaded successfully!")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
        tmp_audio.write(uploaded_file.read())
        tmp_audio_path = tmp_audio.name

    st.audio(uploaded_file, format="audio/mp3")
    st.write("🔄 Transcribing audio...")

    # Get transcription
    try:
        with open(tmp_audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )
        st.write("✅ Transcription received!")
    except Exception as e:
        st.error(f"❌ Error during transcription: {e}")
        os.remove(tmp_audio_path)
        st.stop()

    st.write("Processing transcription...")

    # Sentence-based Segmentation with Empty Word Handling
    def collect_uppercase_sentences(transcription):
        uppercase_sentences = []
        current_sentence = []
        sentence_start_time = None
        last_end_time = None

        for word_data in transcription.words:
            word = word_data.word.strip()  # Remove surrounding spaces
            start_time = word_data.start
            end_time = word_data.end

            # If word is empty, skip it
            if not word:
                continue

            # If a valid end_time exists, update last_end_time
            if end_time is not None:
                last_end_time = end_time

            # Starts with Upper (ignore "I")
            if word[0].isupper() and word != "I":
                # If there's an existing sentence, save it first
                if current_sentence and sentence_start_time is not None:
                    full_sentence = " ".join(current_sentence)
                    uppercase_sentences.append(
                        (sentence_start_time, last_end_time, full_sentence)
                    )

                # New sentence
                current_sentence = [word]
                sentence_start_time = start_time
            else:
                current_sentence.append(word)

        if current_sentence and sentence_start_time is not None:
            full_sentence = " ".join(current_sentence)
            uppercase_sentences.append(
                (sentence_start_time, last_end_time, full_sentence)
            )

        return uppercase_sentences

    uppercase_sentences_list = collect_uppercase_sentences(transcription)

    # Sentiment Analysis
    timestamps = []
    sentiments = []

    for sentence_start, sentence_end, sentence in uppercase_sentences_list:
        sentiment_score = sia.polarity_scores(sentence)["compound"]
        timestamps.append(sentence_start)
        sentiments.append(sentiment_score)

    # Plot Sentiment Analysis
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(timestamps, sentiments, marker="o", linestyle="-", label="Sentiment Score")

    average_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    ax.axhline(
        average_sentiment,
        color="red",
        linestyle="--",
        label=f"Avg Sentiment: {average_sentiment:.2f}",
    )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Sentiment Score")
    ax.set_title("Sentiment Analysis of Lyrics Over Time")
    ax.legend()

    st.pyplot(fig)

    # Cleanup temp file
    os.remove(tmp_audio_path)
