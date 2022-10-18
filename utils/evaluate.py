from utils.metrics import multi_metrics_for_valid
import torch


def evaluate(model, dataloader, criterion):
    epoch_loss = 0.0
    y_preds_tags_array = []
    y_tags_array = []

    model.eval()

    with torch.no_grad():
        for idx, (video, audio, text, attr, labels) in enumerate(dataloader):
            # features = features.unsqueeze(1)

            video = video.to(device)
            text = text.to(device)
            audio = audio.to(device)
            attr = attr.to(device)
            labels = labels.to(device)

            predictions = model(text, audio, video, attention_mask=attr)[0]

            # loss = torch.nn.functional.cross_entropy(predictions, labels)

            # _, y_pred_tags = torch.max(predictions, dim = 1)
            # _, y_tags = torch.max(labels, dim = 1)
            # y_pred_tags = y_pred_tags.cpu().data
            # y_tags = y_tags.cpu().data
            predictions = predictions.cpu().data
            predictions = [predictions[i][0].item() for i in range(len(predictions))]
            labels = labels.cpu().data
            labels = [labels[i].item() for i in range(len(labels))]

            y_preds_tags_array.extend(predictions)
            y_tags_array.extend(labels)

            torch.cuda.empty_cache()

    acc, acc_7, prec, f1, rec, report, matrix = multi_metrics_for_valid_with_confusion(y_preds_tags_array, y_tags_array)

    return epoch_loss / len(dataloader), acc, acc_7, prec, f1, rec, report, matrix
