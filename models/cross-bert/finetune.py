import torch
import numpy as np
from torch import nn

from models.bert import BertLayerNorm


class BertFinetun(nn.Module):
    def __init__(self, config):
        super(BertFinetun, self).__init__()
        self.proj_t = nn.Conv1d(768, 30, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(128, 30, kernel_size=1, padding=0, bias=False)

        ch1 = 32
        k1 = (4, 4, 4)  # 3d kernel size
        s1 = (2, 2, 1)  # 3d strides
        pd1 = (1, 1, 2)  # 3d padding
        self.proj_v_1 = nn.Conv3d(
            in_channels=5,
            out_channels=ch1,
            kernel_size=k1,
            stride=s1,
            padding=pd1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(ch1)
        self.drop_video = nn.Dropout3d(0.1)
        self.lstm = nn.LSTM(
            input_size=6656, hidden_size=30, num_layers=2, batch_first=True
        )

        self.activation = nn.ReLU()
        self.audio_weight_1 = torch.nn.Parameter(
            torch.FloatTensor(1), requires_grad=True
        )
        self.text_weight_1 = torch.nn.Parameter(
            torch.FloatTensor(1), requires_grad=True
        )
        self.video_weight_1 = torch.nn.Parameter(
            torch.FloatTensor(1), requires_grad=True
        )
        self.bias = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.audio_weight_1.data.fill_(1)
        self.text_weight_1.data.fill_(1)
        self.video_weight_1.data.fill_(1)
        self.bias.data.fill_(0)
        self.dropout1 = nn.Dropout(0.3)
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm1 = BertLayerNorm(768)

    def forward(
        self, hidden_states, pooled_output, audio_data, video_data, attention_mask
    ):
        attention_mask = attention_mask.squeeze(1)
        attention_mask_ = attention_mask.permute(0, 2, 1)
        text_data = hidden_states
        text_data = text_data.transpose(1, 2)
        text_data = self.proj_t(text_data)
        text_data = text_data.transpose(1, 2)
        text_data_1 = text_data.reshape(-1).cpu().detach().numpy()
        weights = np.sqrt(np.linalg.norm(text_data_1, ord=2))
        text_data = text_data / weights

        audio_data = audio_data.transpose(1, 2)
        audio_data = self.proj_a(audio_data)
        audio_data = audio_data.transpose(1, 2)

        video_data = self.proj_v_1(video_data)
        video_data = self.bn1(video_data)
        video_data = video_data.reshape((video_data.shape[0], 50, 6656))
        video_data, (_, _) = self.lstm(video_data)

        video_att = torch.matmul(video_data, video_data.transpose(-1, -2))
        video_att = self.activation(video_att)

        text_att = torch.matmul(text_data, text_data.transpose(-1, -2))
        text_att1 = self.activation(text_att)

        audio_att = torch.matmul(audio_data, audio_data.transpose(-1, -2))
        audio_att = self.activation(audio_att)

        audio_weight_1 = self.audio_weight_1
        text_weight_1 = self.text_weight_1
        video_weight_1 = self.video_weight_1
        bias = self.bias

        fusion_att = text_weight_1 * text_att1 + audio_weight_1 * audio_att
        fusion_att = fusion_att + video_weight_1 * video_att + bias

        fusion_att1 = self.activation(fusion_att)
        fusion_att = fusion_att + attention_mask + attention_mask_
        fusion_att = nn.Softmax(dim=-1)(fusion_att)
        fusion_att = self.dropout1(fusion_att)

        fusion_data = torch.matmul(fusion_att, hidden_states)
        fusion_data = fusion_data + hidden_states

        hidden_states_new = self.dense(fusion_data)
        hidden_states_new = self.dropout(hidden_states_new)
        hidden_states_new = self.LayerNorm1(hidden_states_new)
        hidden_states_new = hidden_states_new[:, 0]
        return hidden_states_new, text_att1, fusion_att1
