from util.preprocess.audio import AudioProcessor
from util.preprocess.text import TextProcessor
from util.preprocess.video import VideoProcessor


class Preprocessor:
    def __init__(self):
        self.video = VideoProcessor()
        self.audio = AudioProcessor()
        self.text = TextProcessor()

    def process(self, video_path):
        video_features = self.video.get_features(video_path)
        audio_path = self.audio.extract(video_path)
        audio_features = self.audio.get_features(audio_path)
        text = self.text.extract(audio_path)
        text_features = self.text.get_features(text)

        return (
            video_features,
            audio_features,
            text_features["input_ids"][0],
            text_features["attention_mask"][0],
        )
