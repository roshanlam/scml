from pytube import YouTube, Playlist, Channel
import logging


class YouTubeDownloader:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def download_video(self, url, resolution='highest', file_extension='mp4', output_path='./uploaded_videos'):
        try:
            yt = YouTube(url)
            print(yt.streams[0].title)
            print(yt.streams[0].description)
            if resolution == 'highest':
                video = yt.streams.filter(progressive=True, file_extension=file_extension).order_by('resolution').desc().first()
            else:
                video = yt.streams.filter(progressive=True, file_extension=file_extension, resolution=resolution).first()
            if video:
                return video.download(output_path=output_path)
            else:
                logging.error("No suitable video stream found.")
                return None
        except Exception as e:
            logging.error(f"Failed to download video: {e}")
            return None

    def download_playlist(self, playlist_url, resolution='highest', file_extension='mp4', output_path='./uploaded_videos'):
        try:
            playlist = Playlist(playlist_url)
            file_paths = []
            for video in playlist.videos:
                file_path = self.download_video(video.watch_url, resolution, file_extension, output_path)
                if file_path:
                    file_paths.append(file_path)
                    logging.info(f"Downloaded {video.title} successfully.")
            return file_paths
        except Exception as e:
            logging.error(f"Failed to download playlist: {e}")
            return []

    def fetch_playlist_urls(self, playlist_url):
        try:
            playlist = Playlist(playlist_url)
            return list(playlist.video_urls)
        except Exception as e:
            logging.error(f"Failed to fetch playlist URLs: {e}")
            return []

"""downloader = YouTubeDownloader()
video_url = "http://youtube.com/watch?v=2lAe1cqCOXo"
playlist_url = "https://www.youtube.com/watch?v=GrX4WfT5FI4&list=PL6gx4Cwl9DGDv5eyBLEd9l3ZZzVoroxIZ"

# Download a single video
downloader.download_video(video_url)

# Download all videos in a playlist
downloader.download_playlist(playlist_url)

# Fetch all URLs from a playlist
urls = downloader.fetch_playlist_urls(playlist_url)
for url in urls:
    print(url)"""
