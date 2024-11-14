'''
Speech recognition and Transcribing workflow in Tour
'''

# Import libraries
import speech_recognition as sr
import time
import audioop
from STT import process_audio   


# Initialize the recognizer
recognizer = sr.Recognizer()

def record_audio():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        energy_threshold = recognizer.energy_threshold
        silence_threshold = 0.5 * energy_threshold
        print(silence_threshold)
        audio = None
        silence_start = None
        started_speaking = False

        # (not so good. Dev pending)
        print("Preparing to listen...")
        # Wait for speech to begin 
        while True:
            try:
                # Start recording immediately upon detecting speech
                audio = recognizer.listen(source, timeout = 1) # CHeckout parameters of sr
                started_speaking = True
                print("Listening... please continue speaking.")
                break

            except sr.WaitTimeoutError:
                continue

        # Continue recording until 6 seconds of silence is detected
        silence_start = 0
        while started_speaking:
            try:
                new_audio = recognizer.listen(source, timeout = 1) # Check
                energy = audioop.rms(new_audio.get_raw_data(), 2)

                # Silence check: detect continuous 6 seconds of silence
                if energy < silence_threshold:  # Lowered threshold for stricter silence detection
                    if silence_start == 0:
                        silence_start = time.time()  # Start the silence timer here if it hasn't started yet

                    elif time.time() - silence_start > 6: # Stop recording
                        break
                else:
                    silence_start = None  # Reset silence timer if noise is detected

                # Concatenate audio chunks
                audio = sr.AudioData(audio.get_raw_data() + new_audio.get_raw_data(), audio.sample_rate, audio.sample_width)

            except sr.WaitTimeoutError:
                break

        # Process and transcribe if speaking was detected
        if started_speaking:
            # Check if the audio data is not too short (estimated at 8000 bytes)
            if len(audio.get_raw_data()) < 8000:
                return "Can you please repeat slowly"
            else: # Process audio input
                final_text = process_audio(audio, audio.sample_rate, audio.sample_width)
                if final_text:
                    return final_text
                else:
                    return "Sorry, I couldn't understand that."

if __name__ == '__main__':
    # from tts_run import main
    from rag_agent import query_agent
    #record_audio()
    # response = query_agent(output)
    # main(response)
    #print(time.time() - start)