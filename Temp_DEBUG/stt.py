
import os
from dotenv import load_dotenv
import requests
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from pydub.utils import make_chunks
from concurrent.futures import ThreadPoolExecutor, as_completed
import wave
import json


# # Function to record audio from the microphone
# def record_audio(duration):
#     """
#     Records audio from the microphone for a specified duration.

#     Parameters:
#     duration_ms (int): Duration of the recording in milliseconds.

#     Returns:
#     str: The path to the recorded audio file.
#     """
#     fs = 44100  # Sample rate
#     print("Recording...")
#     audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
#     sd.wait()  # Wait until recording is finished
#     print("Recording finished.")

#     # Save the recorded audio as a WAV file
#     audio_file_path = "recorded_audio.wav"
#     with wave.open(audio_file_path, 'wb') as wf:
#         wf.setnchannels(1)
#         wf.setsampwidth(2)  # 16-bit
#         wf.setframerate(fs)
#         wf.writeframes(audio_data.tobytes())

    return audio_file_path

# Function to send an audio chunk to the API
def send_audio_chunk(file_path):
    """
    Sends an audio chunk to the speech-to-text API and returns the response.

    Parameters:
    file_path (str): The path of the audio chunk file.

    Returns:
    str: The API's response text or an error message.
    """
    url = "https://api.sarvam.ai/speech-to-text-translate"
    load_dotenv()
    api_key = os.getenv("SARVAM_API_KEY")

    if not api_key:
        return f"Error: API key not found for {file_path}."

    headers = {
        "api-subscription-key": api_key,
    }

    try:
        with open(file_path, "rb") as audio_file:
            file_name = os.path.basename(file_path)
            files = {
                'file': (file_name, audio_file, 'audio/wav')  # MIME type is important
            }

            response = requests.post(url, files=files, headers=headers)

            if response.status_code == 200:
                return response.text.strip()  # Return the recognized text
            else:
                return f"Error: {file_path} - Status Code: {response.status_code} - Response: {response.text}"
    except Exception as e:
        return f"Error: {file_path} - An exception occurred: {e}"

# Function to split audio into chunks and process each chunk
def process_audio_in_chunks(input_file, chunk_length_ms=10000):
    """
    Splits an audio file into smaller chunks and processes each chunk using the API.

    Parameters:
    input_file (str): The input audio file to split and process.
    chunk_length_ms (int): Length of each chunk in milliseconds. Default is 10 seconds.

    Returns:
    str: The concatenated transcription result from all chunks.
    """
    audio = AudioSegment.from_wav(input_file)
    os.remove(input_file)  # Remove the original audio file
    chunks = make_chunks(audio, chunk_length_ms)  # Split audio into chunks

    chunk_paths = []
    for i, chunk in enumerate(chunks):
        chunk_name = f"chunk_{i}.wav"
        chunk.export(chunk_name, format="wav")  # Export each chunk as a separate .wav file
        chunk_paths.append(chunk_name)

    # Process each chunk in parallel
    transcriptions = [None] * len(chunk_paths)
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(send_audio_chunk, path): path for path in chunk_paths}

        for future in as_completed(futures):
            try:
                result = future.result()
                index = chunk_paths.index(futures[future])
                transcriptions[index] = result
            except Exception as e:
                print(f"An error occurred while processing {futures[future]}: {e}")

    # Clean up the temporary chunk files
    for chunk_path in chunk_paths:
        os.remove(chunk_path)

    # Join the transcriptions
    transcriptions = [json.loads(transcription)['transcript'] for transcription in transcriptions]
    final_transcription = " ".join(transcriptions)
    return final_transcription

# Example usage
if __name__ == "__main__":
    recording_duration = 5  # Record for 10 seconds
    input_audio_file = record_audio(recording_duration)

    # Process the audio in chunks and get the final transcription
    final_text = process_audio_in_chunks(input_audio_file, chunk_length_ms=10000)  # 10-second chunks
    print("Final Transcription:\n", final_text)
