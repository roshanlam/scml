from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.views import APIView
from rest_framework.generics import ListAPIView
from rest_framework.response import Response
from django.conf import settings
from .audio import preprocess_audio, AudioTranscriber
from .videosutil import YouTubeDownloader
from .serializers import TranscriptionSerializer, FileUploadSerializer
from .models import Transcription
import shutil
import ffmpeg
import os


class DownloadVideoView(APIView):
    def post(self, request):
        url = request.data.get('url')
        if not url:
            return Response({"error": "URL is required"}, status=status.HTTP_400_BAD_REQUEST)

        downloader = YouTubeDownloader()
        file_path = downloader.download_video(url)
        if file_path:
            return Response({"message": "Video downloaded successfully", "file_path": file_path})
        else:
            return Response({"error": "Failed to download video"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class TranscriptionListView(ListAPIView):
    queryset = Transcription.objects.all()
    serializer_class = TranscriptionSerializer


class FileUploadView(APIView):
    # Todo: Implement a secure way for the only the main api to access this api
    parser_classes = [MultiPartParser, FormParser]

    @csrf_exempt
    def post(self, request, *args, **kwargs):
        serializer = FileUploadSerializer(data=request.data)
        if serializer.is_valid():
            file_obj = request.FILES['file']
            source_type = request.data.get('source_type')
            source_id = request.data.get('source_id')
            source_title = request.data.get('source_title')
            source_owner = request.data.get('source_owner')

            video_path = os.path.join("uploaded_videos", file_obj.name)
            with open(video_path, 'wb+') as destination:
                for chunk in file_obj.chunks():
                    destination.write(chunk)

            audio_path = os.path.join("converted_audios", f"{os.path.splitext(file_obj.name)[0]}.wav")
            ffmpeg.input(video_path).output(audio_path, acodec='pcm_s16le', ac=1, ar='16000').run(overwrite_output=True)

            transcriber = AudioTranscriber()
            transcription = transcriber.transcribe(audio_path)

            Transcription.objects.create(
                source_type=source_type,
                source_id=source_id,
                source_title=source_title,
                source_owner=source_owner,
                transcription=transcription
            )
            return Response({"message": "File processed successfully", "transcription": transcription})
        else:
            return Response(serializer.errors, status=400)