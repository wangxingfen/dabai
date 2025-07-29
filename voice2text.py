import requests
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import tempfile
import os


def record_audio_vad(sample_rate=44100, silence_threshold=0.01, silence_duration=2):
    """
    Record audio with voice activity detection
    """
    print("Speak now! (Recording will stop automatically after silence)")
    
    # Buffer to store audio chunks
    audio_buffer = []
    silence_counter = 0
    recording_started = False
    
    def audio_callback(indata, frames, time, status):
        nonlocal recording_started, silence_counter
        # Calculate audio amplitude
        audio_data = indata[:, 0]
        amplitude = np.max(np.abs(audio_data))
        
        # Add to buffer
        audio_buffer.append(audio_data.copy())
        
        # Check if voice has started
        if not recording_started and amplitude > silence_threshold:
            recording_started = True
            print("Voice detected...")
            
        # Check for silence to stop recording
        if recording_started:
            if amplitude < silence_threshold:
                silence_counter += 1
            else:
                silence_counter = 0
                
    # Start recording stream
    with sd.InputStream(callback=audio_callback, samplerate=sample_rate, channels=1, dtype=np.float32):
        # Wait for recording to start
        while not recording_started:
            sd.sleep(100)
            
        # Continue until silence duration is reached
        while silence_counter < silence_duration * 10:  # 10 chunks per second approximately
            sd.sleep(100)
            
    print("Recording finished!")
    
    # Convert buffer to numpy array
    audio_data = np.concatenate(audio_buffer)
    
    # Save to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    write(temp_file.name, sample_rate, (audio_data * 32767).astype(np.int16))  # Convert float32 to int16
    return temp_file.name


def record_audio(duration=5, sample_rate=44100):
    """
    Record audio from the default microphone
    """
    print("Recording... Speak now!")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()  # Wait until recording is finished
    print("Recording finished!")
    
    # Save to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    write(temp_file.name, sample_rate, audio_data)
    return temp_file.name


def voice2text(file_path):
    url = "https://api.siliconflow.cn/v1/audio/transcriptions"
    headers = {
        "Authorization": "Bearer sk-aardxirtqpsnhqvocoqblbiirgtoeqmlgldlrzrjondasxll"
    }
    with open(file_path, 'rb') as audio_file:
        files = {
            'model': (None, 'FunAudioLLM/SenseVoiceSmall'),
            'file': ('output.wav', audio_file, 'audio/wav')
        }
        response = requests.post(url, headers=headers, files=files)
        print(response.json()["text"])
        return response.json()["text"]


def record_and_transcribe(duration=5, use_vad=True):
    """
    Record audio from microphone and transcribe it
    """
    # Record audio
    if use_vad:
        temp_file_path = record_audio_vad()
    else:
        temp_file_path = record_audio(duration)
    
    try:
        # Transcribe the recorded audio
        transcription = voice2text(temp_file_path)
        return transcription
    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)


if __name__ == "__main__":
    while True:
        transcription = record_and_transcribe()
        print(f"Transcription: {transcription}")
