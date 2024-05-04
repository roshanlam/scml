from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
import ffmpeg
import os
import shutil
import soundfile as sf
from scipy.signal import resample
import torch
from transformers import pipeline
import csv

app = FastAPI()

UPLOAD_DIRECTORY = "uploaded_videos"
OUTPUT_DIRECTORY = "converted_audios"
CSV_FILE_PATH = "transcription_data.csv"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


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


def save_to_csv(data):
    try:
        with open(CSV_FILE_PATH, "x", newline="") as file:
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
        with open(CSV_FILE_PATH, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data)
    print(f"Data saved to {CSV_FILE_PATH}")


@app.post("/upload/")
async def upload_video(
    file: UploadFile = File(...),
    source_type: str = Form(...),
    source_id: str = Form(...),
    source_title: str = Form(...),
    source_owner: str = Form(...),
):
    if not file.filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Invalid file format")
    video_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    audio_path = os.path.join(
        OUTPUT_DIRECTORY, f"{os.path.splitext(file.filename)[0]}.wav"
    )
    ffmpeg.input(video_path).output(
        audio_path, acodec="pcm_s16le", ac=1, ar="16000"
    ).run(overwrite_output=True)

    transcriber = AudioTranscriber()
    text = transcriber.transcribe(audio_path)
    data = [source_type, source_id, source_title, source_owner, text]
    save_to_csv(data)

    return FileResponse(
        audio_path, media_type="audio/wav", filename=os.path.basename(audio_path)
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
