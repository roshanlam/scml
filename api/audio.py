import torch
import soundfile as sf
from scipy.signal import resample
from transformers import pipeline
import ffmpeg


def convert_video_to_audio(video_path, audio_path):
    try:
        (
            ffmpeg.input(video_path)
            .output(
                audio_path, acodec="pcm_s16le", ac=1, ar="16k"
            )  # WAV format, mono channel, 16kHz
            .run(overwrite_output=True)
        )
        print(f"Audio extracted and saved to {audio_path}")
    except ffmpeg.Error as e:
        print("An error occurred while converting the video to audio:", e)


# The following code does the job for now but is not so good so we need to use openai's whisper which is better
def preprocess_audio(audio_path, target_sample_rate=16000):
    audio_data, sample_rate = sf.read(audio_path)
    if sample_rate != target_sample_rate:
        audio_data = resample(
            audio_data, int(len(audio_data) * target_sample_rate / sample_rate)
        )
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    return audio_data, target_sample_rate


class AudioTranscriber:
    def __init__(
        self,
        model_name="facebook/wav2vec2-base-960h",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.transcription_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=0 if device == "cuda" else -1,
        )

    def transcribe(self, audio_path, chunk_size=10):
        raw_audio, sample_rate = preprocess_audio(audio_path, target_sample_rate=16000)
        results = []
        for start in range(0, len(raw_audio), sample_rate * chunk_size):
            end = start + sample_rate * chunk_size
            chunk = raw_audio[start:end]
            if len(chunk) == 0:
                continue
            result = self.transcription_pipeline(chunk, sampling_rate=sample_rate)
            results.append(result["text"])
        return " ".join(results)


def save_to_csv(self, data, file_path):
    try:
        with open(file_path, "x", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "source_type",
                    "source_id",
                    "source_title",
                    "source_owner",
                    "transcription",
                ]
            )
            writer.writerow(data)
    except FileExistsError:
        with open(file_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data)
        print(f"Data saved to {file_path}")


"""if __name__ == "__main__":
    transcriber = AudioTranscriber()
    text = transcriber.transcribe("data/LTM.wav")
    print("Transcription:", text)
    data = [
        "video",
        "001",
        "LLM Application Development - Tutorial 5 - Long Term Memory",
        "thenewboston",
        text,
    ]
    transcriber.save_to_csv(data, "transcription_data.csv")"""
