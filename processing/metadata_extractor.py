# processing/metadata_extractor.py
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from urllib.parse import urlparse
from loguru import logger
import spacy
from textblob import TextBlob
from dateutil import parser as date_parser

class MetadataExtractor:
    """Extracts metadata from news articles"""
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.spacy_model_name = spacy_model
        self.nlp = None
        self._load_spacy_model()
        
        # Compile regex patterns
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'),
            'currency': re.compile(r'\$[\d,]+\.?\d*|\d+\.\d+\s*(USD|EUR|GBP|dollars?|euros?|pounds?)'),
            'percentage': re.compile(r'\d+\.?\d*\s*%'),
            'date_mentions': re.compile(r'\b(today|yesterday|tomorrow|this week|next week|last week|this month|next month|last month|this year|next year|last year)\b', re.IGNORECASE),
            'time_mentions': re.compile(r'\b\d{1,2}:\d{2}\s*(AM|PM|am|pm)?\b'),
            'numbers': re.compile(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'),
            'quotes': re.compile(r'"([^"]*)"'),
            'social_handles': re.compile(r'@\w+'),
            'hashtags': re.compile(r'#\w+'),
            'urls': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        }
    
    def _load_spacy_model(self):
        """Load spaCy model for NLP processing"""
        try:
            self.nlp = spacy.load(self.spacy_model_name)
            logger.info(f"Loaded spaCy model: {self.spacy_model_name}")
        except OSError:
            logger.error(f"spaCy model '{self.spacy_model_name}' not found. Please install it with: python -m spacy download {self.spacy_model_name}")
            self.nlp = None
    
    def extract_reading_time(self, content: str, wpm: int = 200) -> int:
        """Estimate reading time in minutes"""
        if not content:
            return 0
        
        word_count = len(content.split())
        reading_time = max(1, round(word_count / wpm))
        return reading_time
    
    def extract_language(self, content: str) -> str:
        """Detect content language using TextBlob"""
        try:
            blob = TextBlob(content[:1000])  # Use first 1000 chars for detection
            return blob.detect_language()
        except:
            return "en"  # Default to English
    
    def extract_keywords(self, content: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords using spaCy NLP"""
        if not self.nlp or not content:
            return []
        
        try:
            doc = self.nlp(content)
            
            # Extract keywords based on POS tags and frequency
            keywords = []
            word_freq = {}
            
            for token in doc:
                # Skip stop words, punctuation, and short words
                if (token.is_stop or token.is_punct or len(token.text) < 3 or 
                    token.pos_ in ['PRON', 'DET', 'AUX', 'CONJ', 'ADP']):
                    continue
                
                # Focus on nouns, adjectives, and proper nouns
                if token.pos_ in ['NOUN', 'ADJ', 'PROPN']:
                    word = token.lemma_.lower()
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Sort by frequency and return top keywords
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            keywords = [word for word, freq in sorted_words[:max_keywords]]
            
            return keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract named entities using spaCy"""
        if not self.nlp or not content:
            return {}
        
        try:
            doc = self.nlp(content)
            entities = {}
            
            for ent in doc.ents:
                entity_type = ent.label_
                entity_text = ent.text.strip()
                
                if entity_type not in entities:
                    entities[entity_type] = []
                
                if entity_text not in entities[entity_type]:
                    entities[entity_type].append(entity_text)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return {}
    
    def extract_quotes(self, content: str) -> List[str]:
        """Extract quoted text from content"""
        if not content:
            return []
        
        quotes = self.patterns['quotes'].findall(content)
        # Filter out short quotes (likely not meaningful)
        meaningful_quotes = [quote for quote in quotes if len(quote) > 20]
        return meaningful_quotes
    
    def extract_numbers_and_stats(self, content: str) -> Dict[str, List[str]]:
        """Extract numbers, percentages, and currency mentions"""
        if not content:
            return {}
        
        stats = {
            'numbers': self.patterns['numbers'].findall(content),
            'percentages': self.patterns['percentage'].findall(content),
            'currency': self.patterns['currency'].findall(content)
        }
        
        return stats
    
    def extract_temporal_references(self, content: str) -> List[str]:
        """Extract temporal references from content"""
        if not content:
            return []
        
        temporal_refs = []
        
        # Date mentions
        date_mentions = self.patterns['date_mentions'].findall(content)
        temporal_refs.extend(date_mentions)
        
        # Time mentions
        time_mentions = self.patterns['time_mentions'].findall(content)
        temporal_refs.extend(time_mentions)
        
        return temporal_refs
    
    def extract_social_media_references(self, content: str) -> Dict[str, List[str]]:
        """Extract social media handles and hashtags"""
        if not content:
            return {}
        
        return {
            'handles': self.patterns['social_handles'].findall(content),
            'hashtags': self.patterns['hashtags'].findall(content)
        }
    
    def extract_contact_info(self, content: str) -> Dict[str, List[str]]:
        """Extract contact information"""
        if not content:
            return {}
        
        return {
            'emails': self.patterns['email'].findall(content),
            'phones': self.patterns['phone'].findall(content)
        }
    
    def extract_article_structure(self, content: str) -> Dict[str, Any]:
        """Analyze article structure"""
        if not content:
            return {}
        
        paragraphs = content.split('\n\n')
        sentences = content.split('.')
        
        # Analyze paragraph lengths
        paragraph_lengths = [len(p.split()) for p in paragraphs if p.strip()]
        
        # Analyze sentence lengths
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        
        structure = {
            'paragraph_count': len(paragraph_lengths),
            'sentence_count': len(sentence_lengths),
            'avg_paragraph_length': sum(paragraph_lengths) / len(paragraph_lengths) if paragraph_lengths else 0,
            'avg_sentence_length': sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0,
            'longest_paragraph': max(paragraph_lengths) if paragraph_lengths else 0,
            'shortest_paragraph': min(paragraph_lengths) if paragraph_lengths else 0
        }
        
        return structure
    
    def extract_url_metadata(self, url: str) -> Dict[str, str]:
        """Extract metadata from article URL"""
        if not url:
            return {}
        
        try:
            parsed = urlparse(url)
            
            # Extract domain information
            domain_parts = parsed.netloc.split('.')
            
            metadata = {
                'domain': parsed.netloc,
                'path': parsed.path,
                'scheme': parsed.scheme,
                'subdomain': domain_parts[0] if len(domain_parts) > 2 else '',
                'tld': domain_parts[-1] if domain_parts else ''
            }
            
            # Try to extract date from URL path
            date_match = re.search(r'/(\d{4})/(\d{1,2})/(\d{1,2})/', parsed.path)
            if date_match:
                try:
                    year, month, day = date_match.groups()
                    metadata['url_date'] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                except:
                    pass
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting URL metadata: {str(e)}")
            return {}
    
    def extract_comprehensive_metadata(self, title: str, content: str, url: str = "", 
                                     author: str = "", published: datetime = None) -> Dict[str, Any]:
        """Extract comprehensive metadata from article"""
        try:
            metadata = {
                'basic_info': {
                    'title': title,
                    'author': author,
                    'published': published.isoformat() if published else None,
                    'url': url,
                    'language': self.extract_language(content),
                    'reading_time': self.extract_reading_time(content)
                },
                'content_analysis': {
                    'word_count': len(content.split()) if content else 0,
                    'character_count': len(content) if content else 0,
                    'keywords': self.extract_keywords(content),
                    'structure': self.extract_article_structure(content)
                },
                'entities': self.extract_entities(content),
                'quotes': self.extract_quotes(content),
                'numbers_and_stats': self.extract_numbers_and_stats(content),
                'temporal_references': self.extract_temporal_references(content),
                'social_media': self.extract_social_media_references(content),
                'contact_info': self.extract_contact_info(content),
                'url_metadata': self.extract_url_metadata(url)
            }
            
            # Add extraction timestamp
            metadata['extraction_timestamp'] = datetime.now(timezone.utc).isoformat()
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting comprehensive metadata: {str(e)}")
            return {}
    
    def extract_summary_metadata(self, content: str) -> Dict[str, Any]:
        """Extract lightweight metadata for quick processing"""
        try:
            return {
                'word_count': len(content.split()) if content else 0,
                'reading_time': self.extract_reading_time(content),
                'language': self.extract_language(content),
                'has_quotes': bool(self.patterns['quotes'].search(content)),
                'has_numbers': bool(self.patterns['numbers'].search(content)),
                'has_dates': bool(self.patterns['date_mentions'].search(content)),
                'extraction_timestamp': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Error extracting summary metadata: {str(e)}")
            return {}

# Utility functions
def test_metadata_extractor():
    """Test function for the metadata extractor"""
    extractor = MetadataExtractor()
    
    test_content = """
    Apple Inc. announced today that CEO Tim Cook will speak at the conference tomorrow at 2:30 PM.
    The company's stock rose 5.2% to $150.25 after the announcement.
    "We're excited about our new product lineup," Cook said in a statement.
    The event will be held in Cupertino, California.
    Contact us at info@apple.com or call (408) 996-1010.
    #AppleEvent @tim_cook
    """
    
    metadata = extractor.extract_comprehensive_metadata(
        title="Apple CEO to Speak at Conference",
        content=test_content,
        url="https://news.example.com/2024/01/15/apple-ceo-conference",
        author="John Doe"
    )
    
    print("Extracted metadata:")
    import json
    print(json.dumps(metadata, indent=2, default=str))

if __name__ == "__main__":
    test_metadata_extractor()