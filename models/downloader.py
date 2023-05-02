import torch
from models.intermodal_fusion.finetune import BertFinetun as intermodalBERT
from models.cross_bert.finetune import BertFinetun as crossBERT
from models.bert import BertConfig
from models.classifier import BertForSequenceClassification


class Downloader:
    def get_model(self, config_path, source_path, num_labels, device, base_model_name='intermodal'):
        config = BertConfig.from_json_file(config_path)
        bert_finetun = get_base_model(base_model_name)
        model = BertForSequenceClassification(config=config, num_labels=num_labels, bert_finetun=bert_finetun)

        model.load_state_dict(torch.load(source_path))
        model = model.to(device)

        return model


def get_base_model(name):
    if name == 'intermodal':
        return intermodalBERT
    elif name == 'cross':
        return crossBERT
    else:
        raise NameError('no such model')
