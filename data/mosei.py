import pandas as pd
from cv2 import resize
from torch.utils.data import Dataset


class MoseiDataset(Dataset):
    labels = ["sadness", "happiness", "anger", "disgust"]

    def __init__(self, annotations_file):
        self.data = []

        data = pd.read_pickle(annotations_file)

        ltoi = {
            "sadness": 0,
            "happiness": 1,
            "anger": 2,
            "disgust": 3,
        }

        for item in data:
            video_features = item[0]
            video_features = [
                resize(img, dsize=(64, 64)) for img in video_features if img != []
            ]
            audio_features = item[1]
            text_features = item[2]
            emotion = item[3]
            emotion = ltoi[emotion]

            self.data.append((text_features, audio_features, video_features, emotion))

        del data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        elem = self.data[idx]
        #      text,    audio,   video,    sentiment
        return elem[0], elem[1], elem[2], elem[3]
