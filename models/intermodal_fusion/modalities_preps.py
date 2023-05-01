from torch.nn import Conv1d, Conv3d
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import BatchNorm3d
from torch.nn import Dropout3d
from torch.nn import LSTM
import numpy as np
import torch


class TextPrep(Module):
    def __init__(self):
        super(TextPrep, self).__init__()
        self.proj_t = Conv1d(768, 30, kernel_size=1, padding=0, bias=False)
        self.activation = ReLU()

    def forward(self, data):
        text_data = data
        text_data = text_data.transpose(1, 2)
        text_data = self.proj_t(text_data)
        text_data = text_data.transpose(1, 2)
        text_data_1 = text_data.reshape(-1).cpu().detach().numpy()
        weights = np.sqrt(np.linalg.norm(text_data_1, ord=2))
        text_data = text_data / weights

        text_att = torch.matmul(text_data, text_data.transpose(-1, -2))
        text_att = self.activation(text_att)

        return text_att


class AudioPrep(Module):
    def __init__(self):
        super(AudioPrep, self).__init__()
        self.proj_a = Conv1d(128, 30, kernel_size=1, padding=0, bias=False)
        self.activation = ReLU()

    def forward(self, data):
        audio_data = data
        audio_data = audio_data.transpose(1, 2)
        audio_data = self.proj_a(audio_data)
        audio_data = audio_data.transpose(1, 2)

        audio_att = torch.matmul(audio_data, audio_data.transpose(-1, -2))
        audio_att = self.activation(audio_att)

        return audio_att


class VideoPrep(Module):
    def __init__(self):
        super(VideoPrep, self).__init__()
        ch1 = 32
        k1 = (4, 4, 4)  # 3d kernel size
        s1 = (2, 2, 1)  # 3d strides
        pd1 = (1, 1, 2)  # 3d padding
        self.proj_v_1 = Conv3d(in_channels=5, out_channels=ch1,
                               kernel_size=k1, stride=s1,
                               padding=pd1, bias=False)
        self.bn1 = BatchNorm3d(ch1)
        self.drop_video = Dropout3d(0.1)
        self.lstm = LSTM(input_size=6656, hidden_size=30,
                         num_layers=2, batch_first=True)
        self.activation = ReLU()

    def forward(self, data):
        video_data = data
        video_data = self.proj_v_1(video_data)
        video_data = self.bn1(video_data)

        video_data = video_data.reshape((video_data.shape[0], 50, 6656))
        video_data, (_, _) = self.lstm(video_data)

        video_att = torch.matmul(video_data, video_data.transpose(-1, -2))
        video_att = self.activation(video_att)

        return video_att
