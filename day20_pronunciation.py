
import os
import json
import re
import time
import sounddevice as sd

from scipy.io.wavfile import write
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
from groq import Groq


# Configuration
MODEL = "whisper-large-v3-turbo"
SAMPLE_RATE = 16000
RECORDING_DURATION = 15

AUDIO_FILE = Path("day20_recording.wav")
OUTPUT_FILE = Path("day20_assessment.json")

REFERENCE_TEXT = (
    "Hi team, my name is Ravi Teja. Nice to meet you. "
    "Today, I am practicing my English speaking and pronunciation. "
    "I will speak clearly and confidently."
)


# Initialize Groq client
def initialize_groq():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Check your .env file."
        )

    return Groq(api_key=api_key)


# Record audio from the microphone
def record_audio():
    try:
        print(f"\nRecording duration: {RECORDING_DURATION} seconds")
        print("Recording starts in 3 seconds...")
        time.sleep(3)

        print("Recording... Read the reference text aloud!")

        audio = sd.rec(
            int(RECORDING_DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16"
        )

        sd.wait()

        write(str(AUDIO_FILE), SAMPLE_RATE, audio)

        print("\nAudio saved to:", AUDIO_FILE.resolve())
        return True

    except Exception as error:
        print("Recording failed:", error)
        print("Check your microphone and Windows permissions.")
        return False


# Normalize text for comparison
def normalize_words(text):
    return re.findall(r"\b[\w']+\b", text.lower())


# Compare reference and transcript using word-level edit distance
def compare_text(reference, transcript):
    expected = normalize_words(reference)
    actual = normalize_words(transcript)

    if not expected:
        raise ValueError("Reference text cannot be empty.")

    rows = len(expected) + 1
    cols = len(actual) + 1

    dp = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dp[i][0] = i

    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = (
                0
                if expected[i - 1] == actual[j - 1]
                else 1
            )

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    errors = dp[-1][-1]

    accuracy = max(
        0,
        1 - errors / len(expected)
    ) * 100

    return {
        "reference_word_count": len(expected),
        "transcript_word_count": len(actual),
        "word_errors": errors,
        "word_match_percentage": round(accuracy, 2)
    }


# Transcribe the saved audio using Groq
def transcribe_audio(client):
    with AUDIO_FILE.open("rb") as audio:
        result = client.audio.transcriptions.create(
            file=(AUDIO_FILE.name, audio.read()),
            model=MODEL,
            response_format="json"
        )

    return result.text.strip()


# Save assessment results to JSON
def save_results(transcript, metrics):
    result = {
        "task": "Day 20 - Pronunciation Assessment",
        "provider": "Groq",
        "model": MODEL,
        "audio_file": str(AUDIO_FILE.resolve()),
        "reference_text": REFERENCE_TEXT,
        "transcription": transcript,
        "assessment": metrics,
        "processed_at": datetime.now(
            timezone.utc
        ).isoformat()
    }

    OUTPUT_FILE.write_text(
        json.dumps(result, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\nJSON saved to:", OUTPUT_FILE.resolve())


# Main program
def main():
    try:
        client = initialize_groq()
        print("Groq client initialized successfully.")

        print("\n--- Reference Text ---")
        print(REFERENCE_TEXT)
        print("\nRead this sentence when recording starts.")

        input("\nPress Enter when you are ready to record...")

        if not record_audio():
            return

        print("\nTranscribing recorded audio...")
        transcript = transcribe_audio(client)

        metrics = compare_text(
            REFERENCE_TEXT,
            transcript
        )

        print("\n--- Assessment Result ---")
        print("Reference:", REFERENCE_TEXT)
        print("Transcript:", transcript)
        print("Word errors:", metrics["word_errors"])
        print(
            "Word match:",
            metrics["word_match_percentage"],
            "%"
        )
        print("Audio saved to:", AUDIO_FILE.resolve())

        save_results(transcript, metrics)

    except Exception as error:
        print("\nProcessing failed:", error)


if __name__ == "__main__":
    main()
