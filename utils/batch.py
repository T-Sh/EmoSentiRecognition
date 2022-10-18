from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
import numpy as np

max_l = 25
group_size = 10
overlap = 5

BATCH_SIZE = 24
TEXT_MAX_LENGTH = 50

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def collate_batch(batch):
    label_list, video_list = [], []
    text_list, att_masks = [], []
    audio_list = []

    for (_text_features, _audio_features, _video_features, _label) in batch:
        global max_l

        label_list.append(_label)

        float_f = []

        ind = 0
        while ind < max_l:
            group = []
            end_pos = ind + overlap

            for i in range(ind, end_pos):
                if i >= len(_video_features):
                    break
                f = _video_features[i]
                if f != []:
                    f = rgb2gray(f)
                else:
                    f = [[0 for col in range(64)] for row in range(64)]
                f = torch.FloatTensor(f)

                group.append(f)

            for i in range(len(group), group_size):
                group.append(torch.zeros(64, 64))

            group = torch.transpose(pad_sequence(group), 0, 1)
            float_f.append(group)
            ind += overlap

        try:
            X = torch.transpose(pad_sequence(float_f), 0, 1)
        except Exception:
            print(len(float_f), float_f[0].shape, float_f[1].shape)
            raise Exception

        pt = tokenizer(_text_features, padding="max_length", add_special_tokens=True, max_length=TEXT_MAX_LENGTH,
                       return_tensors="pt")

        text_list.append(pt["input_ids"][0][:TEXT_MAX_LENGTH])
        att_masks.append(pt["attention_mask"][0][:TEXT_MAX_LENGTH])
        audio_list.append(_audio_features)
        video_list.append(X)

    video_features_tensor = torch.transpose(pad_sequence(video_list), 0, 1)
    audio_tensor = torch.FloatTensor(audio_list)
    audio_tensor = torch.unsqueeze(audio_tensor, 1)
    text_tensor = torch.stack(text_list)
    att_tensor = torch.stack(att_masks)
    label_tensor = torch.FloatTensor(label_list)

    return video_features_tensor, audio_tensor, text_tensor, att_tensor, label_tensor


def make_loaders(train_dataset, valid_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, collate_fn=collate_batch)

    return train_dataloader, valid_dataloader
