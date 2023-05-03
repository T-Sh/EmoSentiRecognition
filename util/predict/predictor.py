from torch import cuda, device

from models.downloader import Downloader
from util.preprocess.preprocessor import Preprocessor


class Predictor:
    def __init__(self, model_download_path):
        self.preprocessor = Preprocessor()
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.downloader = Downloader(model_download_path)
        self.model = self.downloader.get_model(device)

    def predict(self, video_path, labels=None):
        vf, af, tf, mf = self.preprocessor.process(video_path)

        vf = vf.to(self.device)
        af = af.to(self.device)
        tf = tf.to(self.device)
        mf = mf.to(self.device)

        result = self.model([(vf, af, tf, mf)])[0]

        if labels:
            max_value = max(result)
            return result.index(max_value)

        return result
