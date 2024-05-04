from django.db import models

class Transcription(models.Model):
    source_type = models.CharField(max_length=100)
    source_id = models.CharField(max_length=100)
    source_title = models.CharField(max_length=255)
    source_owner = models.CharField(max_length=100)
    transcription = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.source_title} by {self.source_owner}"
