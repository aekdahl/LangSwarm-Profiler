from langswarm.profiler.extract.features import FeatureExtractor

def test_extract_length_of_prompt():
    extractor = FeatureExtractor()
    text = "What is artificial intelligence?"
    result = extractor._extract_length_of_prompt(text)
    assert result == 5  # 5 words in the text

def test_extract_sentiment(mocker):
    extractor = FeatureExtractor()
    mocker.patch.object(
        extractor.sentiment_pipeline,
        "__call__",
        return_value=[{"label": "POSITIVE", "score": 0.98}]
    )
    sentiment = extractor._extract_sentiment("This is great!")
    assert sentiment == "POSITIVE"
