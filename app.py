import yt_dlp
import ffmpeg
import os
import time
import json
import numpy as np
import soundfile as sf
import subprocess
import sys
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
from flask import Flask
import requests
import webrtcvad
import numpy as np
import wave

OUTPUT_FILE = "recognized_texts.txt"
TRANSLATED_FILE = "translated_texts.txt"
TTS_API_URL = "http://10.10.16.34:7006/generate/santali"
TTS_OUTPUT_DIR = "TTS_output"
REM_AUDIO_DIR = "rem_audio"
LOG_FILE = "error_log.txt"
TIMESTAMPS_FILE = "timestamps.txt"

app = Flask(__name__)

os.makedirs('videos', exist_ok=True)
os.makedirs('audios', exist_ok=True)
os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)
os.makedirs(REM_AUDIO_DIR, exist_ok=True)

# Initialize Triton client
triton_client = httpclient.InferenceServerClient(url="10.10.16.34:8011", concurrency=4)

class VideoAudioExtractor:
    def __init__(self, youtube_url, buffer_seconds=5, max_segments=5):
        self.youtube_url = youtube_url
        self.buffer_seconds = buffer_seconds
        self.max_segments = max_segments
        self.stream_url = self._get_stream_url()
    
    def _get_stream_url(self):
        ydl_opts = {'format': 'best', 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(self.youtube_url, download=False)
            return info['url']
    
    def extract_video_audio(self):
        segment_count = 1
        while segment_count <= self.max_segments:
            video_path = os.path.join('videos', f'segment_{segment_count}.mp4')
            
            ffmpeg_video_command = [
                "ffmpeg",
                "-re",
                "-i", self.stream_url,
                "-t", "10",
                "-c:v", "copy",
                "-c:a", "aac",
                "-y",
                video_path
            ]
            subprocess.run(ffmpeg_video_command)
            print(f"Saved video segment {segment_count} as {video_path}.")

            audio_path = self._extract_audio(video_path)
            if audio_path:
                self.run_vad_and_save_timestamps(audio_path, segment_count)
                self.send_audio_to_asr(audio_path, segment_count)
            segment_count += 1

    def _extract_audio(self, video_path):
        audio_path = os.path.join('audios', os.path.basename(video_path).replace('.mp4', '.wav'))
        ffmpeg_audio_command = [
            "ffmpeg",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            audio_path
        ]
        subprocess.run(ffmpeg_audio_command)
        print(f"Saved audio segment as {audio_path}.")
        return audio_path

    def run_vad_and_save_timestamps(self, audio_file_path, segment_count):
        vad = webrtcvad.Vad(1)  # Set aggressiveness level (0-3; 1 is generally a good balance)

        # Parameters for VAD processing
        sample_rate = 16000
        frame_duration_ms = 30
        frame_size = int(sample_rate * frame_duration_ms / 1000)  # Calculate frame size

        # Prepare the output directory
        vad_segments_dir = "vad_segments"
        os.makedirs(vad_segments_dir, exist_ok=True)

        timestamps = []
        with wave.open(audio_file_path, 'rb') as wf:
            audio_data = wf.readframes(wf.getnframes())
            audio_samples = np.frombuffer(audio_data, dtype=np.int16)

            # Process frames and detect speech
            start_time = None
            for i in range(0, len(audio_samples), frame_size):
                frame = audio_samples[i:i + frame_size]
                if len(frame) < frame_size:
                    frame = np.pad(frame, (0, frame_size - len(frame)), 'constant')

                is_speech = vad.is_speech(frame.tobytes(), sample_rate)
                if is_speech:
                    if start_time is None:
                        start_time = i / sample_rate
                else:
                    if start_time is not None:
                        end_time = i / sample_rate
                        timestamps.append((start_time, end_time))
                        start_time = None

            # If still in speech at the end, close the last segment
            if start_time is not None:
                timestamps.append((start_time, len(audio_samples) / sample_rate))

        # Write the timestamps to the file and extract audio segments
        with open("timestamps.txt", "a") as ts_file:
            for idx, (start, end) in enumerate(timestamps, start=1):
                ts_file.write(f"Segment {segment_count}-{idx}: {start:.2f}-{end:.2f} seconds\n")
                # Extract and save each audio segment
                self.save_audio_segment(audio_samples, start, end, sample_rate, vad_segments_dir, segment_count, idx)
        print(f"VAD timestamps and segments for segment {segment_count} saved to timestamps.txt and {vad_segments_dir}.")
        self.send_vad_segments_to_asr(segment_count)

    def save_audio_segment(self,audio_samples, start_time, end_time, sample_rate, output_dir, segment_count, segment_idx):
        # Convert start and end times to sample indices
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        segment_samples = audio_samples[start_sample:end_sample]

        # Define the segment file path
        segment_file_path = os.path.join(output_dir, f"segment_{segment_count}_{segment_idx}.wav")

        # Save the segment as a WAV file
        with wave.open(segment_file_path, 'wb') as segment_file:
            segment_file.setnchannels(1)  # Mono audio
            segment_file.setsampwidth(2)  # 2 bytes per sample for int16
            segment_file.setframerate(sample_rate)
            segment_file.writeframes(segment_samples.tobytes())
        print(f"Audio segment saved: {segment_file_path}")

    def asr_inference(self, audio):
        language = "hi"  # Fixed language
        source_data = np.array(["test string"], dtype="object")
        inputs = [
            httpclient.InferInput("INPUT_AUDIO", audio.shape, np_to_triton_dtype(audio.dtype)),
            httpclient.InferInput("INPUT_LANGUAGE_ID", source_data.shape, np_to_triton_dtype(source_data.dtype))
        ]
        inputs[0].set_data_from_numpy(audio)
        inputs[1].set_data_from_numpy(np.array([language], dtype="object"))
        outputs = [httpclient.InferRequestedOutput("OUTPUT_RECOGNIZED_TEXT")]

        # Run inference
        result = triton_client.infer(model_name="asr", inputs=inputs, outputs=outputs)
        return result.as_numpy("OUTPUT_RECOGNIZED_TEXT")[0].decode('utf-8')

    def send_audio_to_asr(self, audio_file_path, segment_count):
        audio_data, _ = sf.read(audio_file_path, dtype="float32")
        audio_array = np.array(audio_data, dtype='float32')
        
        with open(OUTPUT_FILE, "a", encoding="utf-8") as output_file, open(LOG_FILE, "a", encoding="utf-8") as log_file:
            try:
                # Perform ASR inference using Triton client
                transcript = self.asr_inference(audio_array)
                
                print(f"Recognized text: {transcript}")
                output_file.write(f"{transcript}\n")
                output_file.flush()
                
                # Proceed with translation and TTS
                time.sleep(3)
                self.translate_and_save(transcript, segment_count)

            except Exception as e:
                print(f"Exception occurred for {audio_file_path}: {e}")
                log_file.write(f"Exception for segment {segment_count}: {str(e)}\n")

    def send_vad_segments_to_asr(self, segment_count):
        vad_segments_dir = "vad_segments"
        vad_asr_output_dir = "vad_asr_output"
        os.makedirs(vad_asr_output_dir, exist_ok=True)  # Ensure output directory exists

        output_file_path = os.path.join(vad_asr_output_dir, f"segment_{segment_count}_rec.txt")  # File to store ASR outputs for this segment count

        # Open the output file for appending
        with open(output_file_path, "a", encoding="utf-8") as output_file:
            # Loop through each VAD segment file for the current segment_count
            for vad_file in os.listdir(vad_segments_dir):
                if vad_file.startswith(f"segment_{segment_count}_") and vad_file.endswith(".wav"):
                    vad_file_path = os.path.join(vad_segments_dir, vad_file)

                    try:
                        # Load the audio data for ASR inference
                        audio_data, _ = sf.read(vad_file_path, dtype="float32")
                        audio_array = np.array(audio_data, dtype='float32')

                        # Perform ASR inference
                        transcript = self.asr_inference(audio_array)
                        print(f"ASR output for {vad_file}: {transcript}")

                        # Write the ASR output to the segment-specific file
                        output_file.write(f"{vad_file}: {transcript}\n")
                        output_file.flush()  # Ensure it's written immediately
                    except Exception as e:
                        print(f"Error processing {vad_file} for ASR: {e}")
                        with open(LOG_FILE, "a", encoding="utf-8") as log_file:
                            log_file.write(f"ASR error for {vad_file}: {str(e)}\n")

        print(f"ASR outputs for segment {segment_count} saved to {output_file_path}.")

    def translate_and_save(self, text, segment_count):
        translation_url = 'https://ssmt.iiit.ac.in/onemt'
        translation_body = {'text': text, 'source_language': 'hin', 'target_language': 'sat'}
        try:
            print(f"Sending text to translation API: {text}")
            response = requests.post(translation_url, json=translation_body, verify=False, timeout=10)
            
            if response.status_code == 200:
                response_data = response.json()
                translated_text = response_data.get('data', '')
                
                if translated_text:
                    print(f"Translation API returned translated text: {translated_text}")
                    with open(TRANSLATED_FILE, "a", encoding="utf-8") as translated_file:
                        translated_file.write(f"{translated_text}\n")
                    print(f"Translated text saved to {TRANSLATED_FILE}.")
                    
                    # Proceed to TTS with the translated text
                    self.send_to_tts(translated_text, segment_count)
                else:
                    print("Translation API returned empty data.")
                    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
                        log_file.write(f"Translation API returned empty data for segment {segment_count}.\n")
            else:
                print(f"Translation API returned status {response.status_code}: {response.text}")
                with open(LOG_FILE, "a", encoding="utf-8") as log_file:
                    log_file.write(f"Translation API returned status {response.status_code} for segment {segment_count}: {response.text}\n")
                    
        except Exception as e:
            print(f"Translation error for segment {segment_count}: {e}")
            with open(LOG_FILE, "a", encoding="utf-8") as log_file:
                log_file.write(f"Translation error for segment {segment_count}: {str(e)}\n")
    
    # Modify the send_to_tts function in the main script
    def send_to_tts(self, text, segment_count):
        tts_params = {'text': text, 'gender': 'male'}
        try:
            print(f"Sending text to TTS API: {text}")
            response = requests.post(TTS_API_URL, params=tts_params, verify=False, stream=True)
            
            if response.status_code == 200:
                tts_output_path = os.path.join(TTS_OUTPUT_DIR, f'segment_{segment_count}.wav')
                with open(tts_output_path, "wb") as tts_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        tts_file.write(chunk)
                print(f"TTS output saved: {tts_output_path}")
                subprocess.Popen([sys.executable, "video_processing.py"])

            else:
                print(f"TTS API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"TTS error for segment {segment_count}: {e}")

@app.route('/start_extraction', methods=['GET'])
def start_extraction():
    youtube_url = "https://www.youtube.com/live/WkIpEoAFV9w?feature=shared"
    extractor = VideoAudioExtractor(youtube_url)
    extractor.extract_video_audio()
    return "Started video/audio extraction."

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7000)
