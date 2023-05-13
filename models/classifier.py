from torch.nn import Dropout, Linear, Softmax

from models.bert import BertModel, BertPreTrainedModel


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, bert_finetun):
        super(BertForSequenceClassification, self).__init__(config)
        self.bert = BertModel(config)
        self.BertFinetun = bert_finetun(config)
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.classifier = Linear(config.hidden_size, config.num_labels)
        self.apply(self.init_bert_weights)
        self.labels = config.labels

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
        pooled_output, fusion_att = self.BertFinetun(
            encoder_lastoutput,
            pooled_output,
            all_audio_data,
            video_data,
            extend_mask,
        )
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            probs = Softmax(logits)
            max_prob = max(probs[0])
            index = probs.index(max_prob)
            return labels[index]
        elif self.labels is not None:
            probs = Softmax(logits).dim
            index = probs.argmax(1)[0].item()
            return self.labels[index]
        else:
            return logits, fusion_att
