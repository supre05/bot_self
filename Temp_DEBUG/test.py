import sounddevice as sd
import numpy as np
import wave
import datetime

# Global variable to hold the output file and control flag
output_file = None
recording = True  # Flag to control the recording state

# Silence threshold settings
SILENCE_THRESHOLD = 250  # Adjust as needed
SILENCE_DURATION = 5  # Silence duration in seconds

# Timer for silence detection
silence_counter = 0

def callback(indata, frames, time, status):
    global silence_counter, recording

    if status:
        print(status)

    # Check for speech using a threshold
    if np.abs(indata).mean() > SILENCE_THRESHOLD:
        output_file.writeframes(indata)
        silence_counter = 0  # Reset silence counter
    else:
        silence_counter += frames / 44100  # Increment silence counter in seconds

        # Stop recording if silence exceeds the specified duration
        if silence_counter > SILENCE_DURATION:
            print("Silence detected for too long, stopping recording...")
            recording = False  # Set the flag to stop recording
            raise sd.CallbackStop()

def record_audio():
    global output_file, recording
    filename = f"output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    
    output_file = wave.open(filename, 'wb')
    output_file.setnchannels(1)  # Mono
    output_file.setsampwidth(2)   # 16-bit
    output_file.setframerate(44100)  # Sample rate

    # Start the input stream
    with sd.InputStream(callback=callback, channels=1, dtype='int16'):
        print("Listening for speech...")
        while recording:  # Continue while the recording flag is True
            sd.sleep(100)  # This can be a short sleep, keeping the loop active

    output_file.close()
    print(f"Recording saved as {filename}")

if __name__ == "__main__":
    record_audio()
