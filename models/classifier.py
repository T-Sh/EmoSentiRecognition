from models.bert import BertPreTrainedModel
from models.bert import BertModel
from torch.nn import Dropout
from torch.nn import Linear


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels, bert_finetun):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.BertFinetun = bert_finetun(config)
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.classifier = Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids,
        all_audio_data,
        video_data,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
    ):
        encoder_lastoutput, pooled_output, extend_mask = self.bert(
            input_ids,
            all_audio_data,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=True,
        )
        pooled_output = self.dropout(pooled_output)
        pooled_output, text_att, fusion_att = self.BertFinetun(
            encoder_lastoutput, pooled_output, all_audio_data, video_data, extend_mask
        )
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss = 0.5 * (logits.view(-1) - labels) ** 2
            return loss
        else:
            return logits, text_att, fusion_att
