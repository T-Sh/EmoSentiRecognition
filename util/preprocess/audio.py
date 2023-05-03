import librosa
import moviepy.editor as mp
import opensmile


class AudioProcessor:
    sample_path = "tmp/converted_sample.wav"

    def __init__(self):
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    def extract(self, video_path) -> str:
        clip = mp.VideoFileClip(video_path)
        clip.audio.write_audiofile(self.sample_path)

        return self.sample_path

    def get_features(self, audio_path):
        y, sr = librosa.load(audio_path)
        mfcc_40 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        gemap = self.smile.process_file(audio_path).values.tolist()

        return mfcc_40 + gemap
