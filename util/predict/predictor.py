from util.preprocess.preprocessor import Preprocessor
from torch import device, cuda
from models.downloader import Downloader


class Predictor:
    def __init__(self, config_path, source_path, num_labels, base_model_name):
        self.preprocessor = Preprocessor()
        self.device = device('cuda' if cuda.is_available() else 'cpu')
        self.model = Downloader.get_model(config_path, source_path, num_labels, device, base_model_name)

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
