from flask_cors import CORS
from flask import Flask, flash, request, redirect
from werkzeug.utils import secure_filename
import os
import __main__

from models.intermodal_fusion.finetune import (BertFinetun, Fusion,
                                               TextPrep, AudioPrep, VideoPrep)
from util.predict.predictor import Predictor

from models.classifier import BertForSequenceClassification
from models.bert import (BertModel, BertPreTrainedModel,
                         BertEmbeddings, BertLayerNorm, BertEncoder,
                         BertLayer, BertAttention, BertSelfAttention,
                         BertSelfOutput, BertIntermediate, BertOutput,
                         BertPooler, BertConfig, gelu)

app = Flask(__name__)

__main__.BertForSequenceClassification = BertForSequenceClassification
__main__.BertModel = BertModel
__main__.BertPreTrainedModel = BertPreTrainedModel
__main__.BertEmbeddings = BertEmbeddings
__main__.BertLayerNorm = BertLayerNorm
__main__.BertEncoder = BertEncoder
__main__.BertLayer = BertLayer
__main__.BertAttention = BertAttention
__main__.BertSelfAttention = BertSelfAttention
__main__.BertSelfOutput = BertSelfOutput
__main__.BertIntermediate = BertIntermediate
__main__.BertOutput = BertOutput
__main__.BertPooler = BertPooler
__main__.BertConfig = BertConfig
__main__.gelu = gelu

__main__.BertFinetun = BertFinetun
__main__.Fusion = Fusion
__main__.TextPrep = TextPrep
__main__.AudioPrep = AudioPrep
__main__.VideoPrep = VideoPrep

# Cross Origin Resource Sharing (CORS) handling
CORS(app, resources={'/video': {"origins": "http://localhost:5000"}})
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

model_name = 'Tatyana/iemocap_intermodal_6_emotions'
predictor = Predictor(model_name)


@app.route('/video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        file.save(os.path.join("/tmp/", filename))
        flash('Video successfully uploaded')
        filename = "/tmp/" + filename
        label = predictor.predict(filename)
        return label


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
