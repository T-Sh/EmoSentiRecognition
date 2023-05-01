import opensmile
import librosa


def preprocess_audio(audio_path):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    y, sr = librosa.load(audio_path)
    mfcc_40 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    gemap = smile.process_file(audio_path).values.tolist()

    return mfcc_40 + gemap
