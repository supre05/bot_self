'''
Python script for the STT API endpoint
'''

# Import libraries
import os
import requests
from pydub import AudioSegment
from pydub.utils import make_chunks
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from dotenv import load_dotenv

# Env
load_dotenv()

# API endpoint and headers
api_key = os.getenv("SARVAM_API_KEY")
url = "https://api.sarvam.ai/speech-to-text-translate"
headers = {"api-subscription-key": api_key}

# Functions

def stt_model(audio_bytes): # API endpoint
    files = {'file': ('audio.wav', audio_bytes, 'audio/wav')}
    try:
        response = requests.post(url, files = files, headers=headers, timeout = 20) # Timeout at 60 sec
        response.raise_for_status()

        result = response.json()
        transcript = result.get("transcript", "")
        return transcript
    
    except requests.exceptions.HTTPError as http_err:
        # logger.error(f"HTTP error occurred: {http_err}")  # Log specific HTTP error
        return ""
    
    except requests.exceptions.RequestException as req_err:
        # logger.error(f"Error during API call: {req_err}")  # General request error
        return ""

# Process chunkwise for speed

def chunk_audio(audio_data, sample_rate, sample_width, chunk_length_ms = 10000):
    # Process audio data in chunks and concatenate transcriptions into a single string.
    audio = AudioSegment(
        data = audio_data.get_raw_data(),
        sample_width = sample_width,
        frame_rate = sample_rate,
        channels = 1
    )

    # Split audio into chunks
    return make_chunks(audio, chunk_length_ms)

def transcribe_chunk(chunk):
    with BytesIO() as chunk_bytes:
        chunk.export(chunk_bytes, format="wav")
        chunk_bytes.seek(0)
        return stt_model(chunk_bytes)
    

def process_audio(audio_data, sample_rate, sample_width, chunk_length_ms = 10000):
    # Split audio into chunks
    chunks = chunk_audio(audio_data, sample_rate, sample_width, chunk_length_ms)
    transcriptions = [None] * len(chunks)

    # Use ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Submit tasks to threads
        futures = {executor.submit(transcribe_chunk, chunk): idx for idx, chunk in enumerate(chunks) if chunk.dBFS > -50}

        # Process completed futures as they finish
        for future in as_completed(futures):
            idx = futures[future]
            transcription = future.result()
            if transcription:  # Only add non-empty transcriptions
                transcriptions[idx] = transcription

    # Combine all transcriptions into a single string
    return " ".join(transcriptions)

if __name__ == "__main__":  
    # Debug 
    audio_data = ...  # Placeholder for actual AudioData instance
    sample_rate = 44100
    sample_width = 2
    final_transcription = process_audio(audio_data, sample_rate, sample_width)
    print("Final Transcription:", final_transcription)