# agents/sentiment_agent.py
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
import torch
from loguru import logger
from datetime import datetime
import re
from collections import defaultdict

class SentimentAgent:
    """Sentiment analysis agent for news articles"""
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self.sentiment_pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Sentiment mapping
        self.sentiment_mapping = {
            "LABEL_0": "negative",
            "LABEL_1": "neutral", 
            "LABEL_2": "positive",
            "NEGATIVE": "negative",
            "NEUTRAL": "neutral",
            "POSITIVE": "positive"
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load the sentiment analysis model"""
        try:
            logger.info(f"Loading sentiment model: {self.model_name}")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            logger.info("Sentiment model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading sentiment model: {str(e)}")
            self.sentiment_pipeline = None
    
    def analyze_with_transformer(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using transformer model"""
        try:
            if not self.sentiment_pipeline:
                return {"sentiment": "neutral", "confidence": 0.0, "scores": {}}
            
            # Truncate text if too long
            if len(text) > 512:
                text = text[:512]
            
            # Get sentiment scores
            results = self.sentiment_pipeline(text)[0]
            
            # Process results
            scores = {}
            max_score = 0.0
            predicted_sentiment = "neutral"
            
            for result in results:
                label = result["label"]
                score = result["score"]
                
                # Map label to sentiment
                sentiment = self.sentiment_mapping.get(label, label.lower())
                scores[sentiment] = score
                
                if score > max_score:
                    max_score = score
                    predicted_sentiment = sentiment
            
            return {
                "sentiment": predicted_sentiment,
                "confidence": max_score,
                "scores": scores,
                "method": "transformer"
            }
            
        except Exception as e:
            logger.error(f"Error in transformer sentiment analysis: {str(e)}")
            return {"sentiment": "neutral", "confidence": 0.0, "scores": {}}
    
    def analyze_with_textblob(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Map polarity to sentiment
            if polarity > 0.1:
                sentiment = "positive"
            elif polarity < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            # Calculate confidence based on absolute polarity
            confidence = abs(polarity)
            
            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "polarity": polarity,
                "subjectivity": subjectivity,
                "method": "textblob"
            }
            
        except Exception as e:
            logger.error(f"Error in TextBlob sentiment analysis: {str(e)}")
            return {"sentiment": "neutral", "confidence": 0.0, "polarity": 0.0, "subjectivity": 0.0}
    
    def analyze_sentence_sentiment(self, text: str) -> List[Dict[str, Any]]:
        """Analyze sentiment for individual sentences"""
        try:
            sentences = re.split(r'[.!?]+', text)
            sentence_sentiments = []
            
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if len(sentence) > 10:  # Skip very short sentences
                    sentiment_result = self.analyze_with_transformer(sentence)
                    sentiment_result['sentence_index'] = i
                    sentiment_result['sentence_text'] = sentence
                    sentence_sentiments.append(sentiment_result)
            
            return sentence_sentiments
            
        except Exception as e:
            logger.error(f"Error in sentence sentiment analysis: {str(e)}")
            return []
    
    def analyze_article_sentiment(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment for complete article"""
        try:
            title = article_data.get("title", "")
            content = article_data.get("content", "")
            
            results = {
                "article_id": article_data.get("id", ""),
                "title_sentiment": {},
                "content_sentiment": {},
                "overall_sentiment": {},
                "sentence_sentiments": [],
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            # Analyze title sentiment
            if title:
                results["title_sentiment"] = self.analyze_with_transformer(title)
            
            # Analyze content sentiment
            if content:
                # Overall content sentiment
                results["content_sentiment"] = self.analyze_with_transformer(content)
                
                # Sentence-level sentiment
                results["sentence_sentiments"] = self.analyze_sentence_sentiment(content)
            
            # Calculate overall sentiment (weighted combination)
            results["overall_sentiment"] = self._calculate_overall_sentiment(
                results["title_sentiment"], 
                results["content_sentiment"]
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing article sentiment: {str(e)}")
            return {}
    
    def _calculate_overall_sentiment(self, title_sentiment: Dict, content_sentiment: Dict) -> Dict[str, Any]:
        """Calculate overall sentiment from title and content"""
        try:
            # Weight: 30% title, 70% content
            title_weight = 0.3
            content_weight = 0.7
            
            # Get sentiment scores
            title_scores = title_sentiment.get("scores", {})
            content_scores = content_sentiment.get("scores", {})
            
            # Calculate weighted scores
            overall_scores = {}
            for sentiment in ["positive", "negative", "neutral"]:
                title_score = title_scores.get(sentiment, 0.0)
                content_score = content_scores.get(sentiment, 0.0)
                
                overall_scores[sentiment] = (title_weight * title_score + 
                                           content_weight * content_score)
            
            # Find dominant sentiment
            dominant_sentiment = max(overall_scores, key=overall_scores.get)
            confidence = overall_scores[dominant_sentiment]
            
            return {
                "sentiment": dominant_sentiment,
                "confidence": confidence,
                "scores": overall_scores,
                "method": "weighted_combination"
            }
            
        except Exception as e:
            logger.error(f"Error calculating overall sentiment: {str(e)}")
            return {"sentiment": "neutral", "confidence": 0.0, "scores": {}}
    
    def analyze_batch(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze sentiment for multiple articles"""
        try:
            results = []
            
            for article in articles:
                result = self.analyze_article_sentiment(article)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch sentiment analysis: {str(e)}")
            return []
    
    def get_sentiment_distribution(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get sentiment distribution across articles"""
        try:
            distribution = {"positive": 0, "negative": 0, "neutral": 0}
            confidence_scores = []
            
            for article in articles:
                result = self.analyze_article_sentiment(article)
                overall_sentiment = result.get("overall_sentiment", {})
                
                sentiment = overall_sentiment.get("sentiment", "neutral")
                confidence = overall_sentiment.get("confidence", 0.0)
                
                distribution[sentiment] += 1
                confidence_scores.append(confidence)
            
            # Calculate statistics
            total_articles = len(articles)
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
            return {
                "distribution": distribution,
                "percentages": {
                    sentiment: (count / total_articles) * 100 
                    for sentiment, count in distribution.items()
                },
                "total_articles": total_articles,
                "average_confidence": avg_confidence
            }
            
        except Exception as e:
            logger.error(f"Error calculating sentiment distribution: {str(e)}")
            return {}
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get sentiment agent statistics"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "model_available": self.sentiment_pipeline is not None,
            "sentiment_categories": ["positive", "negative", "neutral"]
        }

# Utility functions
def test_sentiment_agent():
    """Test function for the sentiment agent"""
    agent = SentimentAgent()
    
    test_articles = [
        {
            "id": "test_1",
            "title": "Amazing breakthrough in medical research",
            "content": "Scientists have made an incredible discovery that could revolutionize healthcare. This breakthrough brings hope to millions of patients worldwide."
        },
        {
            "id": "test_2", 
            "title": "Economic crisis deepens",
            "content": "The economic situation continues to deteriorate with rising unemployment and falling stock prices. Many businesses are struggling to survive."
        },
        {
            "id": "test_3",
            "title": "Weather update for tomorrow",
            "content": "Tomorrow's weather will be partly cloudy with temperatures around 20 degrees. Light rain is expected in the evening."
        }
    ]
    
    print("Testing Sentiment Agent...")
    
    # Test individual article analysis
    for article in test_articles:
        print(f"\nAnalyzing: {article['title']}")
        result = agent.analyze_article_sentiment(article)
        
        overall = result.get("overall_sentiment", {})
        print(f"  Overall sentiment: {overall.get('sentiment', 'unknown')}")
        print(f"  Confidence: {overall.get('confidence', 0.0):.3f}")
        
        # Show sentence sentiments
        sentence_sentiments = result.get("sentence_sentiments", [])
        if sentence_sentiments:
            print(f"  Sentence sentiments:")
            for sent in sentence_sentiments[:3]:  # Show first 3
                print(f"    - {sent['sentiment']} ({sent['confidence']:.3f})")
    
    # Test batch analysis
    print(f"\nTesting batch analysis...")
    batch_results = agent.analyze_batch(test_articles)
    print(f"Processed {len(batch_results)} articles")
    
    # Test distribution
    print(f"\nSentiment distribution:")
    distribution = agent.get_sentiment_distribution(test_articles)
    for sentiment, percentage in distribution.get("percentages", {}).items():
        print(f"  {sentiment}: {percentage:.1f}%")
    
    # Show agent stats
    print(f"\nAgent statistics:")
    stats = agent.get_agent_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_sentiment_agent()