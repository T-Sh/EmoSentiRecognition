from torch.utils.data import Dataset
import pandas as pd


class DushaFromFileDataset(Dataset):
    labels = ['angry', 'neutral', 'sad', 'other', 'positive']

    def __init__(self, annotations_file):
        self.data = []

        data = pd.read_pickle(annotations_file)

        ltoi = {
            'angry': 0,
            'neutral': 1,
            'sad': 2,
            'other': 3,
            'positive': 4,
        }

        for item in data:
            audio_features = item[1]
            text_features = item[0]
            emotion = item[2]
            emotion = ltoi[emotion]

            self.data.append((text_features, audio_features, emotion))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        elem = self.data[idx]
        #      text,    audio,   emotion
        return elem[0], elem[1], elem[2]
