import torch


def train(model, dataloader, optimizer, device):
    epoch_loss = 0
    epoch_acc = []
    epoch_prec = []
    epoch_f1 = []
    epoch_rec = []
    report = {}

    model.train()

    for idx, (video, audio, text, attr, labels) in enumerate(dataloader):
        optimizer.zero_grad()

        video = video.to(device)
        text = text.to(device)
        audio = audio.to(device)
        attr = attr.to(device)
        labels = labels.to(device)

        loss = model(text, audio, video, attention_mask=attr, labels=labels)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        torch.cuda.empty_cache()

    return epoch_loss / len(dataloader), epoch_acc, epoch_prec, epoch_f1, epoch_rec, report
