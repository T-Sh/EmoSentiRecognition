import speech_recognition as sr
from transformers import BertTokenizer


class TextProcessor:
    def __init__(self, bert_name="bert-base-uncased", max_length=50):
        self.recognizer = sr.Recognizer()
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)
        self.max_length = max_length

    def extract(self, audio_path) -> str:
        audio = sr.AudioFile(audio_path)

        with audio as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio_file = self.recognizer.record(source)

        result = self.recognizer.recognize_google(audio_file)

        return result

    def get_features(self, text):
        return self.tokenizer(
            text,
            padding="max_length",
            add_special_tokens=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
