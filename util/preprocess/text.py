import speech_recognition as sr
from transformers import BertTokenizer


class TextProcessor:
    def __init__(self, bert_name="bert-base-uncased", max_length=50):
        self.recognizer = sr.Recognizer()
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)
        self.max_length = max_length

    def extract(self, audio_path) -> str:
        with sr.WavFile(audio_path) as source:
            # listen for the data (load audio to memory)
            audio_data = self.recognizer.record(source)
            self.recognizer.adjust_for_ambient_noise(source, duration=5)
            # recognize (convert from speech to text)
            try:
                result = self.recognizer.recognize(audio_data)
            except LookupError:
                result = "empty string"

        return result

    def get_features(self, text):
        return self.tokenizer(
            text,
            padding="max_length",
            add_special_tokens=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
