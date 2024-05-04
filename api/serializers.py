from rest_framework import serializers
from .models import Transcription

class TranscriptionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Transcription
        fields = '__all__'

class FileUploadSerializer(serializers.Serializer):
    file = serializers.FileField(max_length=None, allow_empty_file=False)

    def validate_file(self, value):
        if not value.name.endswith('.wav'):
            raise serializers.ValidationError("Only WAV files are allowed.")
        return value
