import os
import time
import ffmpeg
from moviepy.editor import VideoFileClip, AudioFileClip, vfx

class VideoProcessor:
    REM_AUDIO_DIR = "rem_audio"
    TTS_OUTPUT_DIR = "TTS_output"
    VIDEO_OUTPUT_DIR = "video_output"
    VIDEOS_DIR = "videos"
    ERROR_LOG_FILE = "process_error_log.txt"

    def __init__(self):
        # Ensure necessary directories exist
        os.makedirs(self.REM_AUDIO_DIR, exist_ok=True)
        os.makedirs(self.VIDEO_OUTPUT_DIR, exist_ok=True)

    def log_error(self, message):
        """Log errors with timestamp to an error log file."""
        with open(self.ERROR_LOG_FILE, "a") as log_file:
            log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

    def remove_audio_from_video(self, segment_count):
        """Remove audio from video using reliable function and save in rem_audio directory."""
        input_video_path = os.path.join(self.VIDEOS_DIR, f'segment_{segment_count}.mp4')
        output_video_path = os.path.join(self.REM_AUDIO_DIR, f'segment_{segment_count}.mp4')
        
        try:
            # Call the reliable remove_audio function directly
            self.remove_audio(input_video_path, output_video_path)
            
            # Call add_tts_audio_to_video directly after audio removal
            self.add_tts_audio_to_video(segment_count)
        
        except Exception as e:
            error_message = f"Error removing audio from segment {segment_count}: {e}"
            print(error_message)
            self.log_error(error_message)

    def remove_audio(self, input_video_path, output_video_path):
        """Remove audio from a video file and save the output."""
        try:
            # Check if the input file exists
            if not os.path.isfile(input_video_path):
                print(f"Error: Input video file '{input_video_path}' does not exist.")
                return
            
            # Use ffmpeg to remove audio from the video
            ffmpeg.input(input_video_path).output(output_video_path, **{'an': None}).run(overwrite_output=True)
            print(f"Audio removed successfully. Saved as '{output_video_path}'")
        
        except ffmpeg.Error as e:
            error_message = f"An error occurred while processing the video: {e.stderr.decode('utf8')}"
            print(error_message)
            self.log_error(error_message)

    def add_tts_audio_to_video(self, segment_count):
        """Add TTS audio to video, slowing down video to match audio length if needed."""
        video_path = os.path.join(self.REM_AUDIO_DIR, f'segment_{segment_count}.mp4')
        audio_path = os.path.join(self.TTS_OUTPUT_DIR, f'segment_{segment_count}.wav')
        output_path = os.path.join(self.VIDEO_OUTPUT_DIR, f'segment_{segment_count}.mp4')
        
        try:
            video_clip = VideoFileClip(video_path)
            audio_clip = AudioFileClip(audio_path)

            print(f"Video duration: {video_clip.duration}s, Audio duration: {audio_clip.duration}s")

            if audio_clip.duration <= video_clip.duration:
                # Set audio directly if audio is shorter or equal to video duration
                final_clip = video_clip.set_audio(audio_clip)
            else:
                # Calculate slowdown factor if audio is longer
                slowdown_factor = audio_clip.duration / video_clip.duration
                slowed_video_clip = video_clip.fx(vfx.speedx, 1 / slowdown_factor)
                final_clip = slowed_video_clip.set_audio(audio_clip)
                print(f"Applied slowdown factor: {slowdown_factor}, new video duration: {slowed_video_clip.duration}s")

            final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
            print(f"Combined video and TTS audio saved as: {output_path}")
        
        except Exception as e:
            error_message = f"Error adding TTS audio to segment {segment_count}: {e}"
            print(error_message)
            self.log_error(error_message)

    def process_all_video_segments(self):
        """Process all video segments in sequence, waiting for TTS audio if needed."""
        segment_files = sorted(
            [f for f in os.listdir(self.VIDEOS_DIR) if f.startswith('segment_') and f.endswith('.mp4')],
            key=lambda x: int(x.split('_')[1].split('.')[0])
        )

        for segment_file in segment_files:
            segment_count = int(segment_file.split('_')[1].split('.')[0])
            print(f"Processing segment {segment_count}...")

            # Wait until the TTS audio file for the segment is available
            audio_path = os.path.join(self.TTS_OUTPUT_DIR, f'segment_{segment_count}.wav')
            while not os.path.exists(audio_path):
                print(f"TTS audio for segment {segment_count} not found. Waiting...")
                time.sleep(5)  # Retry every 5 seconds

            # Start processing for this segment by removing audio, which will automatically call add_tts_audio
            self.remove_audio_from_video(segment_count)
            print(f"Finished processing segment {segment_count}")

if __name__ == "__main__":
    processor = VideoProcessor()
    processor.process_all_video_segments()
