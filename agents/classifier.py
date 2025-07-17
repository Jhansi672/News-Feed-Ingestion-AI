# agents/classifier.py
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
import torch
from loguru import logger
import re
from collections import Counter
import pickle
import os

class NewsClassifier:
    """Multi-approach news article classifier"""
    
    def __init__(self, categories: List[str] = None, model_name: str = "facebook/bart-large-mnli"):
        self.categories = categories or [
            "Politics", "Technology", "Business", "Sports", "Entertainment", 
            "Science", "Health", "World News", "Local News", "Opinion"
        ]
        self.model_name = model_name
        self.classifier = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Keyword-based classification
        self.category_keywords = self._build_category_keywords()
        
        # TF-IDF classifier (for when we have training data)
        self.tfidf_vectorizer = None
        self.tfidf_classifier = None
        
        self._initialize_transformer_classifier()
    
    def _build_category_keywords(self) -> Dict[str, List[str]]:
        """Build keyword dictionaries for each category"""
        return {
            "Politics": [
                "government", "election", "vote", "politician", "congress", "senate", 
                "president", "democracy", "policy", "legislation", "campaign", "political",
                "republican", "democrat", "liberal", "conservative", "parliament", "minister"
            ],
            "Technology": [
                "technology", "tech", "software", "hardware", "computer", "internet",
                "artificial intelligence", "ai", "machine learning", "blockchain", "crypto",
                "startup", "innovation", "digital", "cyber", "app", "platform", "algorithm"
            ],
            "Business": [
                "business", "economy", "market", "stock", "finance", "investment", "company",
                "corporate", "revenue", "profit", "earnings", "trade", "commerce", "industry",
                "economic", "financial", "banking", "merger", "acquisition", "ceo", "enterprise"
            ],
            "Sports": [
                "sports", "game", "team", "player", "match", "championship", "tournament",
                "football", "basketball", "baseball", "soccer", "tennis", "golf", "olympics",
                "athlete", "coach", "score", "league", "season", "competition"
            ],
            "Entertainment": [
                "entertainment", "movie", "film", "actor", "actress", "celebrity", "music",
                "concert", "album", "song", "artist", "television", "tv", "show", "series",
                "hollywood", "awards", "oscar", "grammy", "premiere", "performance"
            ],
            "Science": [
                "science", "research", "study", "scientist", "discovery", "experiment",
                "medical", "health", "medicine", "disease", "treatment", "vaccine", "virus",
                "climate", "environment", "space", "nasa", "physics", "chemistry", "biology"
            ],
            "Health": [
                "health", "medical", "doctor", "hospital", "patient", "treatment", "disease",
                "medicine", "healthcare", "wellness", "fitness", "nutrition", "diet", "exercise",
                "mental health", "therapy", "surgery", "diagnosis", "symptoms", "pandemic"
            ],
            "World News": [
                "international", "world", "global", "foreign", "country", "nation", "war",
                "conflict", "peace", "diplomatic", "embassy", "crisis", "refugee", "immigration",
                "terrorism", "security", "united nations", "europe", "asia", "africa"
            ],
            "Local News": [
                "local", "community", "city", "town", "municipal", "mayor", "council",
                "neighborhood", "resident", "regional", "county", "state", "district",
                "school district", "police", "fire department", "traffic", "weather"
            ],
            "Opinion": [
                "opinion", "editorial", "commentary", "analysis", "perspective", "viewpoint",
                "columnist", "op-ed", "letter to editor", "blog", "think", "believe",
                "argue", "suggest", "recommend", "criticism", "review", "debate"
            ]
        }
    
    def _initialize_transformer_classifier(self):
        """Initialize transformer-based zero-shot classifier"""
        try:
            logger.info(f"Loading transformer classifier: {self.model_name}")
            self.classifier = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1
            )
            logger.info("Transformer classifier loaded successfully")
        except Exception as e:
            logger.error(f"Error loading transformer classifier: {str(e)}")
            self.classifier = None
    
    def classify_with_keywords(self, text: str, title: str = "") -> Dict[str, float]:
        """Classify using keyword matching"""
        try:
            # Combine title and text (title gets more weight)
            combined_text = f"{title} {title} {text}".lower()
            
            scores = {}
            total_keywords = 0
            
            for category, keywords in self.category_keywords.items():
                category_score = 0
                
                for keyword in keywords:
                    # Count keyword occurrences
                    count = len(re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', combined_text))
                    category_score += count
                    total_keywords += count
                
                scores[category] = category_score
            
            # Normalize scores
            if total_keywords > 0:
                for category in scores:
                    scores[category] = scores[category] / total_keywords
            
            return scores
            
        except Exception as e:
            logger.error(f"Error in keyword classification: {str(e)}")
            return {category: 0.0 for category in self.categories}
    
    def classify_with_transformer(self, text: str, title: str = "") -> Dict[str, float]:
        """Classify using transformer model"""
        try:
            if not self.classifier:
                logger.warning("Transformer classifier not available")
                return {category: 0.0 for category in self.categories}
            
            # Combine title and text
            combined_text = f"{title}. {text}"
            
            # Truncate if too long
            if len(combined_text) > 1000:
                combined_text = combined_text[:1000]
            
            # Perform classification
            result = self.classifier(combined_text, self.categories)
            
            # Convert to dictionary
            scores = {}
            for label, score in zip(result['labels'], result['scores']):
                scores[label] = score
            
            # Ensure all categories are present
            for category in self.categories:
                if category not in scores:
                    scores[category] = 0.0
            
            return scores
            
        except Exception as e:
            logger.error(f"Error in transformer classification: {str(e)}")
            return {category: 0.0 for category in self.categories}
    
    def classify_with_tfidf(self, text: str, title: str = "") -> Dict[str, float]:
        """Classify using TF-IDF and trained model"""
        try:
            if not self.tfidf_vectorizer or not self.tfidf_classifier:
                logger.warning("TF-IDF classifier not trained")
                return {category: 0.0 for category in self.categories}
            
            combined_text = f"{title} {text}"
            
            # Vectorize text
            text_vector = self.tfidf_vectorizer.transform([combined_text])
            
            # Get prediction probabilities
            probabilities = self.tfidf_classifier.predict_proba(text_vector)[0]
            
            # Map to categories
            scores = {}
            for i, category in enumerate(self.categories):
                scores[category] = probabilities[i] if i < len(probabilities) else 0.0
            
            return scores
            
        except Exception as e:
            logger.error(f"Error in TF-IDF classification: {str(e)}")
            return {category: 0.0 for category in self.categories}
    
    def classify_article(self, content: str, title: str = "", 
                        method: str = "ensemble") -> Dict[str, Any]:
        """Main classification method"""
        try:
            results = {
                'method': method,
                'categories': {},
                'top_category': None,
                'confidence': 0.0,
                'all_scores': {}
            }
            
            if method == "keywords":
                scores = self.classify_with_keywords(content, title)
            elif method == "transformer":
                scores = self.classify_with_transformer(content, title)
            elif method == "tfidf":
                scores = self.classify_with_tfidf(content, title)
            elif method == "ensemble":
                # Combine multiple methods
                keyword_scores = self.classify_with_keywords(content, title)
                transformer_scores = self.classify_with_transformer(content, title)
                
                # Weighted combination
                scores = {}
                for category in self.categories:
                    keyword_score = keyword_scores.get(category, 0.0)
                    transformer_score = transformer_scores.get(category, 0.0)
                    
                    # Weight: 30% keywords, 70% transformer
                    scores[category] = 0.3 * keyword_score + 0.7 * transformer_score
                
                results['all_scores'] = {
                    'keywords': keyword_scores,
                    'transformer': transformer_scores
                }
            else:
                raise ValueError(f"Unknown classification method: {method}")
            
            # Find top category
            if scores:
                top_category = max(scores, key=scores.get)
                results['top_category'] = top_category
                results['confidence'] = scores[top_category]
                results['categories'] = scores
            
            return results
            
        except Exception as e:
            logger.error(f"Error classifying article: {str(e)}")
            return {
                'method': method,
                'categories': {category: 0.0 for category in self.categories},
                'top_category': 'Unknown',
                'confidence': 0.0,
                'all_scores': {}
            }
    
    def train_tfidf_classifier(self, training_data: List[Dict[str, Any]]):
        """Train TF-IDF classifier with labeled data"""
        try:
            texts = []
            labels = []
            
            for item in training_data:
                text = f"{item.get('title', '')} {item.get('content', '')}"
                texts.append(text)
                labels.append(item.get('category', 'Unknown'))
            
            # Initialize and fit TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            X = self.tfidf_vectorizer.fit_transform(texts)
            
            # Train classifier
            self.tfidf_classifier = MultinomialNB()
            self.tfidf_classifier.fit(X, labels)
            
            logger.info(f"TF-IDF classifier trained with {len(training_data)} samples")
            
        except Exception as e:
            logger.error(f"Error training TF-IDF classifier: {str(e)}")
    
    def save_tfidf_model(self, filepath: str):
        """Save TF-IDF model to file"""
        try:
            model_data = {
                'vectorizer': self.tfidf_vectorizer,
                'classifier': self.tfidf_classifier,
                'categories': self.categories
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"TF-IDF model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving TF-IDF model: {str(e)}")
    
    def load_tfidf_model(self, filepath: str):
        """Load TF-IDF model from file"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.tfidf_vectorizer = model_data['vectorizer']
            self.tfidf_classifier = model_data['classifier']
            self.categories = model_data['categories']
            
            logger.info(f"TF-IDF model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading TF-IDF model: {str(e)}")
    
    def get_category_distribution(self, articles: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get category distribution for a list of articles"""
        try:
            distribution = {category: 0 for category in self.categories}
            
            for article in articles:
                result = self.classify_article(
                    article.get('content', ''),
                    article.get('title', '')
                )
                
                top_category = result.get('top_category', 'Unknown')
                if top_category in distribution:
                    distribution[top_category] += 1
                else:
                    distribution['Unknown'] = distribution.get('Unknown', 0) + 1
            
            return distribution
            
        except Exception as e:
            logger.error(f"Error getting category distribution: {str(e)}")
            return {category: 0 for category in self.categories}
    
    def classify_batch(self, articles: List[Dict[str, Any]], 
                      method: str = "ensemble") -> List[Dict[str, Any]]:
        """Classify multiple articles in batch"""
        try:
            results = []
            
            for article in articles:
                result = self.classify_article(
                    article.get('content', ''),
                    article.get('title', ''),
                    method=method
                )
                
                # Add article ID if available
                if 'id' in article:
                    result['article_id'] = article['id']
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch classification: {str(e)}")
            return []
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get classifier statistics"""
        return {
            'categories': self.categories,
            'num_categories': len(self.categories),
            'model_name': self.model_name,
            'device': self.device,
            'transformer_available': self.classifier is not None,
            'tfidf_trained': self.tfidf_classifier is not None,
            'keyword_categories': len(self.category_keywords)
        }

# Utility functions
def test_classifier():
    """Test function for the news classifier"""
    classifier = NewsClassifier()
    
    test_articles = [
        {
            'title': 'Apple Announces New iPhone Features',
            'content': 'Apple Inc. today announced new artificial intelligence features for the iPhone, including advanced machine learning capabilities and improved camera software.'
        },
        {
            'title': 'Local Basketball Team Wins Championship',
            'content': 'The hometown basketball team defeated their rivals 95-87 in the championship game last night, with star player scoring 32 points.'
        },
        {
            'title': 'Government Announces New Economic Policy',
            'content': 'The government today announced a new economic policy aimed at reducing inflation and stimulating growth through targeted investments.'
        }
    ]
    
    print("Testing news classifier...")
    
    for i, article in enumerate(test_articles):
        print(f"\nArticle {i+1}: {article['title']}")
        
        # Test different methods
        for method in ['keywords', 'transformer', 'ensemble']:
            result = classifier.classify_article(
                article['content'], 
                article['title'], 
                method=method
            )
            
            print(f"  {method.capitalize()} method:")
            print(f"    Top category: {result['top_category']}")
            print(f"    Confidence: {result['confidence']:.3f}")
    
    # Test batch classification
    print("\nTesting batch classification...")
    batch_results = classifier.classify_batch(test_articles)
    
    distribution = {}
    for result in batch_results:
        category = result['top_category']
        distribution[category] = distribution.get(category, 0) + 1
    
    print("Category distribution:")
    for category, count in distribution.items():
        print(f"  {category}: {count}")
    
    # Show classifier stats
    print("\nClassifier statistics:")
    stats = classifier.get_classification_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_classifier()