import pandas as pd
from torch.utils.data import Dataset
from cv2 import resize


class MosiDataset(Dataset):
    labels = ["-3", "-2", "-1", "0", "1", "2", "3"]

    def __init__(self, annotations_file):
        self.data = []

        data = pd.read_pickle(annotations_file)

        for item in data:
            text_features = item[0]
            audio_features = item[1]
            video_features = item[2]
            video_features = [
                resize(img, dsize=(64, 64)) for img in video_features if img != []
            ]
            sentiment = item[3]

            self.data.append((text_features, audio_features, video_features, sentiment))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        elem = self.data[idx]
        #      text,    audio,   video,    sentiment
        return elem[0], elem[1], elem[2], elem[3]
