import pandas as pd
from torch.utils.data import Dataset


ltoi = {}
emo_num = 0


class MoseiVideoDataset(Dataset):
    def __init__(self, annotations_file):
        self.data = []

        data = pd.read_pickle(annotations_file)

        global ltoi
        global emo_num

        for item in data:
            video_features = item[0]
            audio_features = item[1]
            text_features = item[2]
            emotion = item[3]
            if emotion not in ltoi:
                ltoi[emotion] = emo_num
                emo_num += 1

            emotion = ltoi[emotion]

            self.data.append((text_features, audio_features, video_features, emotion))

        del data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        elem = self.data[idx]
        #      text,    audio,   video,    sentiment
        return elem[0], elem[1], elem[2], elem[3]


train_dataset = MoseiVideoDataset("mosei_train.pkl")
test_dataset = MoseiVideoDataset("mosei_test.pkl")
labels = ["sadness", "happiness", "anger", "disgust"]
