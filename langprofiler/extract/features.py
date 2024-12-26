# feature_extractor.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from bertopic import BERTopic
import torch
from typing import List, Optional, Dict

class FeatureExtractor:
    def __init__(self, device: str = "cpu"):
        """
        Initializes all necessary models for feature extraction.

        :param device: 'cpu' or 'cuda' for GPU acceleration.
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Initialize Intent Extraction Model (T5)
        self.intent_tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.intent_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(self.device)
        
        # Initialize Sentiment Analysis Pipeline
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if self.device=="cuda" else -1
        )
        
        # Initialize Topic Modeling Model (BERTopic)
        self.topic_model = BERTopic()
        
        # You can initialize more models here as needed

    def extract_features(self, text: str, feature_types: List[str]) -> Dict[str, Optional[str]]:
        """
        Extracts specified features from the given text.

        :param text: The input text.
        :param feature_types: List of feature types to extract ('intent', 'topic', 'sentiment').
        :return: Dictionary of extracted features.
        """
        extracted = {}
        for feature in feature_types:
            feature_lower = feature.lower()
            if feature_lower == "intent":
                extracted["intent"] = self._extract_intent(text)
            elif feature_lower == "topic":
                extracted["topic"] = self._extract_topic([text])[0]  # BERTopic expects a list
            elif feature_lower == "sentiment":
                extracted["sentiment"] = self._extract_sentiment(text)
            else:
                print(f"Unsupported feature type: {feature}")
                extracted[feature_lower] = None
        return extracted

    def _extract_intent(self, text: str, max_length: int = 10, num_beams: int = 4) -> str:
        """
        Extracts intent from the given text using T5.

        :param text: The input text.
        :param max_length: Maximum length of the generated intent.
        :param num_beams: Beam search width.
        :return: Extracted intent.
        """
        prompt = f"Extract the intent of the following query:\n\nQuery: \"{text}\"\n\nIntent:"
        inputs = self.intent_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        outputs = self.intent_model.generate(
            inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        intent = self.intent_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return intent

    def _extract_topic(self, texts: List[str]) -> List[str]:
        """
        Extracts topics from the given list of texts using BERTopic.

        :param texts: List of input texts.
        :return: List of extracted topics.
        """
        topics, _ = self.topic_model.fit_transform(texts)
        # Convert numerical topics to readable labels if necessary
        readable_topics = [self.topic_model.get_topic(topic)[0][0] if topic != -1 else "Unknown" for topic in topics]
        return readable_topics

    def _extract_sentiment(self, text: str) -> str:
        """
        Analyzes sentiment of the given text.

        :param text: The input text.
        :return: Sentiment label.
        """
        result = self.sentiment_pipeline(text)[0]
        return result['label']
