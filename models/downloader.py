import torch
import os
from huggingface_hub import snapshot_download

from models.bert import BertConfig
from models.classifier import BertForSequenceClassification
from models.cross_bert.finetune import BertFinetun as crossBERT
from models.intermodal_fusion.finetune import BertFinetun as intermodalBERT


class Downloader:

    available_models = {
        "iemocap": "Tatyana/iemocap_intermodal_6_emotions",
        "meld": "",
        "mosi": "",
        "mosei": "",
    }

    def __init__(self, model_download_path=None):
        if model_download_path is None:
            return

        self.cache_dir = '/tmp/models/'

        snapshot_download(repo_id=model_download_path,
                          local_dir=self.cache_dir,
                          resume_download=True,
                          etag_timeout=300)

        config_name = 'config.json'
        weights_name = 'pytorch_model.pth'

        self.config_file = os.path.join(self.cache_dir, config_name)
        self.weights_path = os.path.join(self.cache_dir, weights_name)

    def __get_model_path(self, model_name: str):
        model_path = self.available_models.get(model_name, "")

        if model_path == "":
            raise LookupError(f"no path found for chosen model {model_name}")

    def get_model(
        self,
        device,
    ):
        model = torch.load(self.weights_path, map_location=device)
        config = BertConfig.from_json_file(self.config_file)
        model.labels = config.labels

        return model


def get_base_model(name: str):
    if name == "intermodal":
        return intermodalBERT
    elif name == "cross":
        return crossBERT
    else:
        raise NameError(f"no such model {name}")
