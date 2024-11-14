import os
from dotenv import load_dotenv
import requests
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment
import io

# Load environment variables from .env file
load_dotenv()

# For audio generation
url = "https://api.sarvam.ai/text-to-speech"
headers = {
    "api-subscription-key": os.getenv("SARVAM_API_KEY"),
    "Content-Type": "application/json"
}

# Ignore
from functools import wraps
def time_it(func):
    import time
    @wraps(func)
    def wrapper(*args,**kwargs):
        start = time.time()
        result = func(*args,**kwargs)
        print(f'time taken by {func.__name__} is {time.time()-start }')
        return result
    return wrapper

# Function to make API request for text-to-speech
@time_it
def generate_audio(text_chunk):
    
    payload = {
            "inputs": [text_chunk],
            "target_language_code": "hi-IN",
            "enable_preprocessing": True
        } 

    # 5 attempts in case of any issue
    fail = True
    for _ in range(5):
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            response_data = response.json()
            fail = False
            break
    if fail:
        print(f"Error: {response.status_code}, {response.text}")
        return None
    
    if "audios" in response_data and len(response_data["audios"]) > 0:
        return base64.b64decode(response_data["audios"][0])
    else:
        print("Error: No audio generated for the chunk.")
        return None

# Function to split text into chunks
@time_it
def split_text(text, max_chunk_size = 1000): # Sarvam TTS limit: 500 chars /text
    ''' Chunk based on sentence splitting'''
    sentences = [part for part in text.split('.') if part]
    for sentence in sentences:
        if len(sentence) > max_chunk_size:
            raise ValueError("Sentence is too long for a chunk.")
    return sentences

@time_it
def combined_audio(audio_segments):
    # Combine all audio segments
    if audio_segments:
        final_audio = AudioSegment.silent(duration=0)  # Start with silent audio
        for segment in audio_segments:
            if segment is not None:  # Check if segment is not None
                final_audio += segment  # Concatenate audio segments

        output_filename = "./static/output_audio.wav"
        final_audio.export(output_filename, format="wav")
        print(f"Audio saved as {output_filename}")
    else:
        print("No audio segments generated.")
    return final_audio
# Main function to orchestrate the audio generation
@time_it
def get_audio(text):
    
    # text_chunks = split_text(text)  # Split the text into chunks
    text_chunks = [text]
    audio_segments = [None] * len(text_chunks)

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(generate_audio, chunk): chunk for chunk in text_chunks}
        
        for future in as_completed(futures):
            audio_data = future.result()
            chunk = futures[future]
            index = text_chunks.index(chunk)  # Find the index of the chunk

            if audio_data:
                audio_segments[index] = AudioSegment.from_wav(io.BytesIO(audio_data))  # Convert bytes to AudioSegment
    final_audio = combined_audio(audio_segments)
    
    return final_audio

    
if __name__ == "__main__":
    # Testing
    # input = "This is a sample text to be converted to speech. The text will be split into multiple chunks and each chunk will be converted to audio separately."
    input = """Welcome to the Museum of Rights and Freedom Tour!

Hello! I’m Samvid, your robotic tour guide. I’ll be with you every step of the way as we explore this museum dedicated to human rights, freedom, and the powerful stories that shaped our country. Before we start, let me share some tips to make your experience smoother:
"""
    get_audio(input)
