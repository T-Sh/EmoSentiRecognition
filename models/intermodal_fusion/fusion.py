from torch.nn import Parameter
from torch.nn import Module
from torch import FloatTensor


class Fusion(Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.audio_weight = Parameter(FloatTensor(1), requires_grad=True)
        self.text_weight = Parameter(FloatTensor(1), requires_grad=True)
        self.video_weight = Parameter(FloatTensor(1), requires_grad=True)
        self.audio_weight.data.fill_(1)
        self.text_weight.data.fill_(1)
        self.video_weight.data.fill_(1)

        self.wat = Parameter(FloatTensor(1), requires_grad=True)
        self.wav = Parameter(FloatTensor(1), requires_grad=True)
        self.wtv = Parameter(FloatTensor(1), requires_grad=True)
        self.wat.data.fill_(1)
        self.wav.data.fill_(1)
        self.wtv.data.fill_(1)

        self.bias_0 = Parameter(FloatTensor(1), requires_grad=True)
        self.bias_1 = Parameter(FloatTensor(1), requires_grad=True)
        self.bias_2 = Parameter(FloatTensor(1), requires_grad=True)
        self.bias_0.data.fill_(0)
        self.bias_1.data.fill_(0)
        self.bias_2.data.fill_(0)

    def forward(self, text_data, audio_data, video_data):
        t, a, v = text_data, audio_data, video_data

        wa = self.audio_weight
        wt = self.text_weight
        wv = self.video_weight

        at = wt * t + wa * a
        av = wa * a + wv * v
        tv = wt * t + wv * v

        wat = self.wat
        wav = self.wav
        wtv = self.wtv

        atv = wat * at + v
        avt = wav * av + t
        tva = wtv * tv + a

        atav = wat * at + wav * av
        attv = wat * at + wtv * tv
        avtv = wav * av + wtv * tv

        b0 = self.bias_0
        b1 = self.bias_1
        b2 = self.bias_2

        fusion_att_0 = at + av + tv + b0
        fusion_att_1 = atv + avt + tva + b1
        fusion_att_2 = atav + attv + avtv + b2

        fusion_att = fusion_att_0 + fusion_att_1 + fusion_att_2

        return fusion_att
