+import sounddevice as sd^M
+from scipy.io.wavfile import write^M
+import numpy as np^M
+^M
+def test_microphone(duration=5):^M
+    freq = 16000  # Sample rate^M
+    print("Testing microphone...")^M
+^M
+    recording = []^M
+^M
+    '''def play_audio(audio_data):^M
+        print("Playing recorded audio...")^M
+        sd.play(audio_data, samplerate=freq)^M
+        sd.wait() '''^M
+^M
+    def callback(indata, frames, time, status):^M
+        audio_data = indata.flatten()^M
+        audio_data_int16 = (audio_data * 32767).astype(np.int16)^M
+        recording.append(audio_data_int16)^M
+^M
+    with sd.InputStream(samplerate=freq, channels=1, callback=callback):^M
+        sd.sleep(duration * 1000)  # Convert seconds to milliseconds^M
+^M
+    if recording:^M
+        recorded_audio = np.concatenate(recording)^M
+        #play_audio(recording)^M
+        write("test_recording.wav", freq, recorded_audio)^M
+        print("Test recording saved as 'test_recording.wav'")^M
+    else:
+        print("No audio recorded.")^M
+^M
+# Call the function^M
+test_microphone()