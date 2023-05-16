# 3-Modal-Cross-Bert

Model for multimodal multiclass sentiment or emotion recognition.
Based on the Cross-Bert model: https://github.com/thuiar/Cross-Modal-BERT with video features additions.

Used datasets:
* Meld (https://paperswithcode.com/dataset/meld)
* MOSI (https://paperswithcode.com/dataset/multimodal-opinionlevel-sentiment-intensity)
* MOSEI (https://paperswithcode.com/dataset/cmu-mosei)
* IEMOCAP (https://paperswithcode.com/dataset/iemocap)

# Intermodal interaction

Model for multimodal multiclass sentiment or emotion recognition.

Used datasets:
* Meld (https://paperswithcode.com/dataset/meld)
* MOSI (https://paperswithcode.com/dataset/multimodal-opinionlevel-sentiment-intensity)
* MOSEI (https://paperswithcode.com/dataset/cmu-mosei)
* IEMOCAP (https://paperswithcode.com/dataset/iemocap)
* RESD (https://huggingface.co/datasets/Aniemore/resd_annotated)
* Dusha (https://paperswithcode.com/dataset/dusha)

# Run

For Docker container building and running execute next commands:

```
docker build -t bert_model --build-arg MODEL_NAME=iemocap .
docker run -d -p 5000:5000 --name bert_model bert_model
```

To use model execute command:

```
curl -F 'file=@/path/to/file.mp4' -H "Content-Type: multipart/form-data" 127.0.0.1:5000/video -X POST
```

Response contains one string value - emotion or sentiment label.

Available trained models (this names can be specified in MODEL_NAME param):

* iemocap (6 emotions)
* mosei-emo (6 emotions)
* mosei-senti (7 classes of sentiment)
* mosi (7 classes of sentiment)
* meld (7 emotions)
