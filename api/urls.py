from django.urls import path
from .views import FileUploadView, TranscriptionListView, DownloadVideoView

urlpatterns = [
    path('upload/', FileUploadView.as_view(), name='file_upload'),
    path('transcriptions/', TranscriptionListView.as_view(), name='transcription-list'),
    path('download_yt_video/', DownloadVideoView.as_view(), name='download_video')
]
