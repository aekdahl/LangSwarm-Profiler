# feature_extractor.py

from typing import List, Dict, Optional
import spacy
from transformers import pipeline
from textblob import TextBlob
from collections import Counter
import re

class FeatureExtractor:
    """
    Class responsible for extracting various linguistic and structural features from text.
    """
    def __init__(self, device: str = "cpu"):
        """
        Initializes the FeatureExtractor with necessary models and pipelines.
        
        :param device: The device to run models on ('cpu' or 'cuda').
        """
        self.nlp = spacy.load("en_core_web_sm")
        self.sentiment_pipeline = pipeline("sentiment-analysis", device=0 if device == "cuda" else -1)
        self.summarizer = pipeline("summarization", device=0 if device == "cuda" else -1)
        # Add any other necessary initializations here

    def extract_features(self, text: str, feature_types: List[str]) -> Dict[str, Optional[str]]:
        """
        Extracts specified features from the given text.
        
        :param text: The input text to extract features from.
        :param feature_types: A list of feature names to extract.
        :return: A dictionary mapping feature names to their extracted values.
        """
        extracted = {}
        for feature in feature_types:
            feature_lower = feature.lower()
            if feature_lower == "intent":
                extracted["intent"] = self._extract_intent(text)
            elif feature_lower == "sentiment":
                extracted["sentiment"] = self._extract_sentiment(text)
            elif feature_lower == "topic":
                extracted["topic"] = self._extract_topic([text])[0]
            elif feature_lower == "entities":
                extracted["entities"] = self._extract_entities(text)
            elif feature_lower == "summarization":
                extracted["summarization"] = self._extract_summarization(text)
            elif feature_lower == "syntax_complexity":
                extracted["syntax_complexity"] = self._extract_syntax_complexity(text)
            elif feature_lower == "readability_score":
                extracted["readability_score"] = self._extract_readability_score(text)
            elif feature_lower == "key_phrase_extraction":
                extracted["key_phrase_extraction"] = self._extract_key_phrases(text)
            elif feature_lower == "temporal_features":
                extracted["temporal_features"] = self._extract_temporal_features(text)
            elif feature_lower == "length_of_prompt":
                extracted["length_of_prompt"] = self._extract_length_of_prompt(text)
            elif feature_lower == "conciseness":
                extracted["conciseness"] = self._extract_conciseness(text)
            else:
                print(f"Unsupported feature type: {feature}")
                extracted[feature_lower] = None
        return extracted

    def _extract_intent(self, text: str) -> Optional[str]:
        """
        Placeholder method for intent extraction.
        Implement intent classification logic here.
        """
        # Example implementation using TextBlob's sentiment as a placeholder
        # Replace with actual intent extraction logic or model
        blob = TextBlob(text)
        return blob.sentiment.subjectivity  # Placeholder value

    def _extract_sentiment(self, text: str) -> Optional[str]:
        """
        Extracts sentiment from the text using a sentiment analysis pipeline.
        """
        try:
            result = self.sentiment_pipeline(text)[0]
            return result['label']  # e.g., 'POSITIVE', 'NEGATIVE', 'NEUTRAL'
        except Exception as e:
            print(f"Sentiment extraction error: {e}")
            return None

    def _extract_topic(self, texts: List[str]) -> List[Optional[str]]:
        """
        Extracts topic from the text. Placeholder for topic modeling.
        Implement topic extraction logic here.
        """
        # Example implementation using simple keyword matching
        topics = []
        topic_keywords = {
            "machine learning": ["machine learning", "ml", "artificial intelligence", "ai"],
            "finance": ["finance", "investment", "retirement", "savings"],
            "weather": ["weather", "forecast", "rain", "sunny"],
            "entertainment": ["movie", "music", "entertainment", "concert"]
            # Add more topics and keywords as needed
        }
        for text in texts:
            text_lower = text.lower()
            topic_found = "unknown"
            for topic, keywords in topic_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    topic_found = topic
                    break
            topics.append(topic_found)
        return topics

    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extracts named entities from the text using spaCy.
        """
        doc = self.nlp(text)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        return entities

    def _extract_summarization(self, text: str) -> Optional[str]:
        """
        Summarizes the text using a summarization pipeline.
        """
        try:
            summary = self.summarizer(text, max_length=50, min_length=25, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            print(f"Summarization error: {e}")
            return None

    def _extract_syntax_complexity(self, text: str) -> Optional[int]:
        """
        Measures the syntactic complexity of the text.
        Example metric: average sentence length or parse tree depth.
        """
        doc = self.nlp(text)
        if not doc.sents:
            return 0
        total_length = sum(len(sent) for sent in doc.sents)
        avg_length = total_length / len(list(doc.sents))
        return int(avg_length)  # Example: average sentence length

    def _extract_readability_score(self, text: str) -> Optional[float]:
        """
        Calculates the readability score of the text using the Flesch-Kincaid formula.
        """
        try:
            from textstat import flesch_kincaid_grade
            score = flesch_kincaid_grade(text)
            return score
        except ImportError:
            print("textstat library not installed. Install it using 'pip install textstat'")
            return None
        except Exception as e:
            print(f"Readability score extraction error: {e}")
            return None

    def _extract_key_phrases(self, text: str) -> List[str]:
        """
        Extracts key phrases from the text. Placeholder for key phrase extraction.
        Implement key phrase extraction logic here.
        """
        # Example implementation using RAKE (Rapid Automatic Keyword Extraction)
        try:
            from rake_nltk import Rake
            r = Rake()
            r.extract_keywords_from_text(text)
            key_phrases = r.get_ranked_phrases()
            return key_phrases[:5]  # Return top 5 key phrases
        except ImportError:
            print("rake_nltk library not installed. Install it using 'pip install rake-nltk'")
            return []
        except Exception as e:
            print(f"Key phrase extraction error: {e}")
            return []

    def _extract_temporal_features(self, text: str) -> List[str]:
        """
        Extracts temporal expressions from the text.
        """
        # Example implementation using regex to find dates
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # e.g., 12/31/2020
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{1,2}, \d{4}\b'  # e.g., January 1, 2020
        ]
        temporal_features = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            temporal_features.extend(matches)
        return temporal_features

    def _extract_length_of_prompt(self, text: str) -> int:
        """
        Calculates the length of the prompt in words.
        """
        word_count = len(text.split())
        return word_count

    def _extract_conciseness(self, text: str) -> float:
        """
        Measures the conciseness of the prompt.
        Returns a score between 0.0 (verbose) to 1.0 (concise).
        """
        total_words = len(text.split())
        if total_words == 0:
            return 1.0
        # Simple heuristic: proportion of unique words
        unique_words = len(set(text.split()))
        conciseness_score = unique_words / total_words
        return min(conciseness_score, 1.0)
