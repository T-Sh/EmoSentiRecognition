import pandas as pd
from torch.utils.data import Dataset
import cv2

MOSEI_LABELS = ["sadness", "happiness", "anger", "disgust"]


class MoseiDataset(Dataset):
    def __init__(self, annotations_file):
        self.data = []

        data = pd.read_pickle(annotations_file)

        ltoi = {}
        emo_num = 0

        for item in data:
            video_features = item[0]
            video_features = [
                cv2.resize(img, dsize=(64, 64)) for img in video_features if img != []
            ]
            audio_features = item[1]
            text_features = item[2]
            emotion = item[3]
            if emotion not in ltoi:
                ltoi[emotion] = emo_num

            emotion = ltoi[emotion]

            self.data.append((text_features, audio_features, video_features, emotion))

        del data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        elem = self.data[idx]
        #      text,    audio,   video,    sentiment
        return elem[0], elem[1], elem[2], elem[3]
