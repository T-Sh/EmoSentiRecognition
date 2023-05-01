from torch.nn import Module
from torch.nn import ReLU, Softmax
from torch.nn import Dropout
from torch.nn import Linear
from models.bert import BertLayerNorm
from models.intermodal_fusion.fusion import Fusion
from models.intermodal_fusion.modalities_preps import TextPrep, AudioPrep, VideoPrep
import torch


class BertFinetun(Module):
    def __init__(self, config):
        super(BertFinetun, self).__init__()

        self.activation = ReLU()
        self.dropout1 = Dropout(0.3)
        self.dense = Linear(768, 768)
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.LayerNorm1 = BertLayerNorm(768)
        self.fusion = Fusion()

        self.text_prep = TextPrep()
        self.audio_prep = AudioPrep()
        self.video_prep = VideoPrep()

    def forward(self, hidden_states, pooled_output, audio_data, video_data, attention_mask):
        attention_mask = attention_mask.squeeze(1)
        attention_mask_ = attention_mask.permute(0, 2, 1)

        text_att = self.text_prep(hidden_states)
        video_att = self.video_prep(video_data)
        audio_att = self.audio_prep(audio_data)

        fusion_att = self.fusion(text_att, audio_att, video_att)

        fusion_att1 = self.activation(fusion_att)
        fusion_att = fusion_att + attention_mask + attention_mask_
        fusion_att = Softmax(dim=-1)(fusion_att)
        fusion_att = self.dropout1(fusion_att)

        fusion_data = torch.matmul(fusion_att, hidden_states)
        fusion_data = fusion_data + hidden_states

        hidden_states_new = self.dense(fusion_data)
        hidden_states_new = self.dropout(hidden_states_new)
        hidden_states_new = self.LayerNorm1(hidden_states_new)
        hidden_states_new = hidden_states_new[:, 0]

        return hidden_states_new, fusion_att1
