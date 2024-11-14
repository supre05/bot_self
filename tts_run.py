'''
Python script for TTS model + app workflow text-> audio file
'''

# Potential opt
# lru cache from functools
#  different accent feature, speed etc
# 

import requests
import wave
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import os
from dotenv import load_dotenv

load_dotenv()

# Set up
api_key = os.getenv("SARVAM_API_KEY")
url = "https://api.sarvam.ai/text-to-speech"
headers = {
    "api-subscription-key": api_key,
    "Content-Type": "application/json"
}

def tts_model(text_chunk): # API endpoint
    payload = {
        "inputs": [text_chunk],
        "target_language_code": "hi-IN",
        "enable_preprocessing": True,
    }
    # Maximum 5 attempts 20sec timeout each
    num_attempt = 1
    for _ in range(num_attempt):
        response = requests.post(url, json=payload, headers=headers, timeout = 20) # 20sec timeout
        if response.status_code == 200: # Success
            response_data = response.json()
            if "audios" in response_data and len(response_data["audios"]) > 0:
                return response_data["audios"][0]
    
    # logger.error(f"Error: {response.status_code}, {response.text}")
    return None

# Function to split text into chunks
def split_text(text, max_chunk_size=1000):
    sentences = [part.strip() for part in text.split('.') if part]  # Split the text into sentences
    result = []

    for sentence in sentences:
        # If sentence is larger than max_chunk_size, split it
        while len(sentence) > max_chunk_size:
            half_index = len(sentence) // 2
            split_point = sentence.rfind(' ', 0, half_index)
            if split_point == -1:  # If no space, split at the midpoint
                split_point = half_index
            result.append(sentence[:split_point].strip())  # First half
            sentence = sentence[split_point:].strip()  # Second half

        result.append(sentence)  # Add remaining sentence
    return result

# Function to combine audio files in order
def combine_audio_files(audio_data_list):
    """
    Combines multiple audio byte streams into a single WAV file.
    """
    if not audio_data_list:
        raise ValueError("No audio data to combine.")
    
    # Decode each base64-encoded audio segment into raw data
    audio_segments = [base64.b64decode(audio_data) for audio_data in audio_data_list if audio_data]

    # Use the first audio segment as a template for parameters
    first_audio = wave.open(io.BytesIO(audio_segments[0]), 'rb')
    params = first_audio.getparams()  # Get audio parameters

    output_stream = io.BytesIO()
    
    # Combine all audio segments into one file
    with wave.open(output_stream, 'wb') as output:
        output.setparams(params)  # Set audio parameters
        
        for segment in audio_segments:
            # Append each decoded segment to the output
            with wave.open(io.BytesIO(segment), 'rb') as audio:
                output.writeframes(audio.readframes(audio.getnframes()))  # Write frames
    
    output_stream.seek(0)
    return output_stream

# Main function to orchestrate audio generation
def get_audio(text):
    text_chunks = split_text(text)
    audio_data_list = [None] * len(text_chunks)  # Placeholder for ordered audio

    # Threading for speed
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(tts_model, chunk): idx for idx, chunk in enumerate(text_chunks)}

        for future in as_completed(futures):
            idx = futures[future]  # Get index for this future
            audio_data = future.result()

            if audio_data:  # Save result in the correct order
                audio_data_list[idx] = audio_data
    # Combine audio
    combined_audio = combine_audio_files(audio_data_list)
    # return output_file
    return combined_audio
# Example usage
if __name__ == "__main__":
    input_text = "This is a test. The text-to-speech service should process this correctly. Let's ensure everything works well."
    audio_data = get_audio(input_text)
    output_path = "output_audio.wav"  # Set your desired file name
    with open(output_path, "wb") as f:
        f.write(audio_data.getvalue())  # Write the BytesIO content to the file

    print(f"Audio saved to {output_path}")