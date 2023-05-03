from torch.utils.data import Dataset


class RESDFromSourceDataset(Dataset):
    labels = ["ang", "disg", "enthus", "fear", "happy", "neut", "sad"]

    def __init__(self, data):
        self.data = []

        ltoi = {
            "anger": 0,
            "disgust": 1,
            "enthusiasm": 2,
            "fear": 3,
            "happiness": 4,
            "neutral": 5,
            "sadness": 6,
        }

        for item in data.values.tolist():
            audio_features = item[1]
            text_features = item[2]
            emotion = item[0]
            emotion = ltoi[emotion]

            self.data.append((text_features, audio_features, emotion))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        elem = self.data[idx]
        #      text,    audio,   emotion
        return elem[0], elem[1], elem[2]
