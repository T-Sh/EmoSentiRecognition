import torch
import tarfile
import tempfile
import os
from pytorch_pretrained_bert.file_utils import cached_path

from models.bert import BertConfig
from models.classifier import BertForSequenceClassification
from models.cross_bert.finetune import BertFinetun as crossBERT
from models.intermodal_fusion.finetune import BertFinetun as intermodalBERT


class Downloader:
    def __init__(self, model_download_path):
        self.cache_dir = '/tmp/models/cache/'
        resolved_archive_file = cached_path(model_download_path, cache_dir=self.cache_dir)
        # Extract archive to temp dir
        tempdir = tempfile.mkdtemp()
        with tarfile.open(resolved_archive_file, 'r:gz') as archive:
            archive.extractall(tempdir)
        serialization_dir = tempdir

        config_name = 'config.json'
        weights_name = 'pytorch_model.pth'

        self.config_file = os.path.join(serialization_dir, config_name)
        self.weights_path = os.path.join(serialization_dir, weights_name)

    def get_model(
        self,
        device,
    ):
        config = BertConfig.from_json_file(self.config_file)
        bert_finetun = get_base_model("intermodal")
        model = BertForSequenceClassification(
            config=config, bert_finetun=bert_finetun
        )

        model.load_state_dict(torch.load( self.weights_path))
        model = model.to(device)

        return model


def get_base_model(name):
    if name == "intermodal":
        return intermodalBERT
    elif name == "cross":
        return crossBERT
    else:
        raise NameError("no such model")
