import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

MAX_LENGTH = 25
GROUP_SIZE = 10
OVERLAP = 5

BATCH_SIZE = 24
TEXT_MAX_LENGTH = 50

TOKENIZER = None


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def collate_batch_3_modals(batch):
    label_list, video_list = [], []
    text_list, att_masks = [], []
    audio_list = []

    for _text_features, _audio_features, _video_features, _label in batch:
        label_list.append(_label)

        video_groups = []

        start_pos = 0
        while start_pos < MAX_LENGTH:
            video_group = []
            end_pos = start_pos + OVERLAP

            for i in range(start_pos, end_pos):
                if i >= len(_video_features):
                    break

                video_feature = _video_features[i]
                if video_feature:
                    video_feature = rgb2gray(video_feature)
                else:
                    # If video is empty, replace it with zeros.
                    video_feature = [[0 for _ in range(64)] for _ in range(64)]

                video_feature = torch.FloatTensor(video_feature)
                video_group.append(video_feature)

            for i in range(len(video_group), GROUP_SIZE):
                video_group.append(torch.zeros(64, 64))

            video_group = torch.transpose(pad_sequence(video_group), 0, 1)
            video_groups.append(video_group)
            start_pos += OVERLAP

        try:
            video = torch.transpose(pad_sequence(video_groups), 0, 1)
        except Exception:
            print(len(video_groups), video_groups[0].shape, video_groups[1].shape)
            raise Exception

        pt = TOKENIZER(
            _text_features,
            padding="max_length",
            add_special_tokens=True,
            max_length=TEXT_MAX_LENGTH,
            return_tensors="pt",
        )

        text_list.append(pt["input_ids"][0][:TEXT_MAX_LENGTH])
        att_masks.append(pt["attention_mask"][0][:TEXT_MAX_LENGTH])
        audio_list.append(_audio_features)
        video_list.append(video)

    video_features_tensor = torch.transpose(pad_sequence(video_list), 0, 1)
    audio_tensor = torch.FloatTensor(audio_list)
    audio_tensor = torch.unsqueeze(audio_tensor, 1)
    text_tensor = torch.stack(text_list)
    att_tensor = torch.stack(att_masks)
    label_tensor = torch.FloatTensor(label_list)

    return (
        video_features_tensor,
        audio_tensor,
        text_tensor,
        att_tensor,
        label_tensor,
    )


def collate_batch_2_modals(batch):
    label_list = []
    text_list, att_masks = [], []
    audio_list = []

    for _text_features, _audio_features, _label in batch:
        label_list.append(_label)

        pt = TOKENIZER(
            _text_features,
            padding="max_length",
            add_special_tokens=True,
            max_length=TEXT_MAX_LENGTH,
            return_tensors="pt",
        )

        text_list.append(pt["input_ids"][0][:TEXT_MAX_LENGTH])
        att_masks.append(pt["attention_mask"][0][:TEXT_MAX_LENGTH])
        audio_list.append(_audio_features)

    audio_tensor = torch.cat(audio_list, dim=0)
    text_tensor = torch.stack(text_list)
    att_tensor = torch.stack(att_masks)
    label_tensor = torch.LongTensor(label_list)

    return audio_tensor, text_tensor, att_tensor, label_tensor


def make_eng_data_loaders(train_dataset, valid_dataset):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch_3_modals,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch_3_modals,
    )

    return train_dataloader, valid_dataloader


def make_rus_data_loaders(train_dataset, valid_dataset):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch_2_modals,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch_2_modals,
    )

    return train_dataloader, valid_dataloader
