# agents/ner_agent.py
import spacy
from typing import Dict, List, Any, Optional, Set, Tuple
import re
from collections import defaultdict, Counter
from loguru import logger
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

class NERAgent:
    """Named Entity Recognition agent for news articles"""
    
    def __init__(self, spacy_model: str = "en_core_web_sm", 
                 transformer_model: str = "dbmdz/bert-large-cased-finetuned-conll03-english"):
        self.spacy_model_name = spacy_model
        self.transformer_model_name = transformer_model
        self.nlp = None
        self.transformer_ner = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Custom entity patterns
        self.custom_patterns = self._build_custom_patterns()
        
        # Entity post-processing rules
        self.entity_filters = self._build_entity_filters()
        
        self._load_models()
    
    def _load_models(self):
        """Load spaCy and transformer models"""
        try:
            # Load spaCy model
            logger.info(f"Loading spaCy model: {self.spacy_model_name}")
            self.nlp = spacy.load(self.spacy_model_name)
            
            # Add custom patterns to spaCy
            self._add_custom_patterns()
            
            logger.info("spaCy model loaded successfully")
            
            # Load transformer model
            logger.info(f"Loading transformer NER model: {self.transformer_model_name}")
            self.transformer_ner = pipeline(
                "ner",
                model=self.transformer_model_name,
                tokenizer=self.transformer_model_name,
                aggregation_strategy="simple",
                device=0 if self.device == "cuda" else -1
            )
            logger.info("Transformer NER model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading NER models: {str(e)}")
            if "spacy" in str(e).lower():
                logger.error(f"Please install spaCy model: python -m spacy download {self.spacy_model_name}")
    
    def _build_custom_patterns(self) -> Dict[str, List[Dict]]:
        """Build custom entity patterns"""
        return {
            "STOCK_SYMBOL": [
                {"UPPER": True, "LENGTH": {">=": 1, "<=": 5}},
                {"ORTH": {"IN": ["Inc.", "Corp.", "Ltd.", "LLC"]}, "OP": "?"}
            ],
            "CURRENCY": [
                {"ORTH": {"IN": ["$", "€", "£", "¥"]}},
                {"LIKE_NUM": True}
            ],
            "PERCENTAGE": [
                {"LIKE_NUM": True},
                {"ORTH": "%"}
            ],
            "EMAIL": [
                {"LIKE_EMAIL": True}
            ],
            "PHONE": [
                {"SHAPE": {"IN": ["ddd-ddd-dddd", "(ddd) ddd-dddd", "ddd.ddd.dddd"]}}
            ]
        }
    
    def _add_custom_patterns(self):
        """Add custom patterns to spaCy"""
        try:
            if not self.nlp:
                return
            
            # Add entity ruler
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            
            patterns = []
            for entity_type, pattern_list in self.custom_patterns.items():
                for pattern in pattern_list:
                    patterns.append({
                        "label": entity_type,
                        "pattern": pattern
                    })
            
            ruler.add_patterns(patterns)
            
        except Exception as e:
            logger.error(f"Error adding custom patterns: {str(e)}")
    
    def _build_entity_filters(self) -> Dict[str, Dict]:
        """Build entity filtering rules"""
        return {
            "PERSON": {
                "min_length": 2,
                "max_length": 50,
                "exclude_patterns": [
                    r"^(Mr|Mrs|Ms|Dr|Prof)\.?$",
                    r"^(The|A|An)$",
                    r"^\d+$"
                ]
            },
            "ORG": {
                "min_length": 2,
                "max_length": 100,
                "exclude_patterns": [
                    r"^(The|A|An)$",
                    r"^\d+$"
                ]
            },
            "GPE": {  # Geopolitical entities
                "min_length": 2,
                "max_length": 50,
                "exclude_patterns": [
                    r"^(The|A|An)$",
                    r"^\d+$"
                ]
            },
            "MONEY": {
                "min_length": 1,
                "max_length": 20
            },
            "DATE": {
                "min_length": 3,
                "max_length": 30
            }
        }
    
    def _filter_entity(self, entity_text: str, entity_type: str) -> bool:
        """Filter entity based on rules"""
        try:
            if entity_type not in self.entity_filters:
                return True
            
            filters = self.entity_filters[entity_type]
            
            # Check length
            if len(entity_text) < filters.get("min_length", 1):
                return False
            if len(entity_text) > filters.get("max_length", 1000):
                return False
            
            # Check exclude patterns
            exclude_patterns = filters.get("exclude_patterns", [])
            for pattern in exclude_patterns:
                if re.match(pattern, entity_text, re.IGNORECASE):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error filtering entity: {str(e)}")
            return True
    
    def extract_entities_spacy(self, text: str) -> Dict[str, List[Dict]]:
        """Extract entities using spaCy"""
        try:
            if not self.nlp:
                return {}
            
            doc = self.nlp(text)
            entities = defaultdict(list)
            
            for ent in doc.ents:
                entity_text = ent.text.strip()
                entity_type = ent.label_
                
                # Filter entity
                if not self._filter_entity(entity_text, entity_type):
                    continue
                
                entity_info = {
                    "text": entity_text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": 1.0,  # spaCy doesn't provide confidence scores
                    "source": "spacy"
                }
                
                entities[entity_type].append(entity_info)
            
            # Remove duplicates
            for entity_type in entities:
                seen = set()
                filtered_entities = []
                for entity in entities[entity_type]:
                    if entity["text"] not in seen:
                        seen.add(entity["text"])
                        filtered_entities.append(entity)
                entities[entity_type] = filtered_entities
            
            return dict(entities)
            
        except Exception as e:
            logger.error(f"Error extracting entities with spaCy: {str(e)}")
            return {}
    
    def extract_entities_transformer(self, text: str) -> Dict[str, List[Dict]]:
        """Extract entities using transformer model"""
        try:
            if not self.transformer_ner:
                return {}
            
            # Truncate text if too long
            if len(text) > 5000:
                text = text[:5000]
            
            # Extract entities
            results = self.transformer_ner(text)
            entities = defaultdict(list)
            
            for result in results:
                entity_text = result["word"]
                entity_type = result["entity_group"]
                confidence = result["score"]
                start = result["start"]
                end = result["end"]
                
                # Filter entity
                if not self._filter_entity(entity_text, entity_type):
                    continue
                
                entity_info = {
                    "text": entity_text,
                    "start": start,
                    "end": end,
                    "confidence": confidence,
                    "source": "transformer"
                }
                
                entities[entity_type].append(entity_info)
            
            return dict(entities)
            
        except Exception as e:
            logger.error(f"Error extracting entities with transformer: {str(e)}")
            return {}
    
    def extract_custom_entities(self, text: str) -> Dict[str, List[Dict]]:
        """Extract custom entities using regex patterns"""
        try:
            entities = defaultdict(list)
            
            # Email addresses
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            for match in re.finditer(email_pattern, text):
                entities["EMAIL"].append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 1.0,
                    "source": "regex"
                })
            
            # Phone numbers
            phone_pattern = r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'
            for match in re.finditer(phone_pattern, text):
                entities["PHONE"].append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 1.0,
                    "source": "regex"
                })
            
            # URLs
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            for match in re.finditer(url_pattern, text):
                entities["URL"].append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 1.0,
                    "source": "regex"
                })
            
            # Stock symbols (simple pattern)
            stock_pattern = r'\b[A-Z]{1,5}\b(?=\s*(?:stock|shares|trading|NYSE|NASDAQ))'
            for match in re.finditer(stock_pattern, text):
                entities["STOCK_SYMBOL"].append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.8,
                    "source": "regex"
                })
            
            # Currency amounts
            currency_pattern = r'\$[\d,]+\.?\d*|\d+\.\d+\s*(?:USD|EUR|GBP|dollars?|euros?|pounds?)'
            for match in re.finditer(currency_pattern, text, re.IGNORECASE):
                entities["CURRENCY"].append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.9,
                    "source": "regex"
                })
            
            # Percentages
            percentage_pattern = r'\d+\.?\d*\s*%'
            for match in re.finditer(percentage_pattern, text):
                entities["PERCENTAGE"].append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.95,
                    "source": "regex"
                })
            
            return dict(entities)
            
        except Exception as e:
            logger.error(f"Error extracting custom entities: {str(e)}")
            return {}
    
    def merge_entities(self, *entity_dicts: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Merge entities from multiple sources"""
        try:
            merged = defaultdict(list)
            
            for entity_dict in entity_dicts:
                for entity_type, entities in entity_dict.items():
                    merged[entity_type].extend(entities)
            
            # Remove duplicates and sort by confidence
            for entity_type in merged:
                # Remove duplicates based on text
                seen = {}
                filtered_entities = []
                
                for entity in merged[entity_type]:
                    text = entity["text"]
                    if text not in seen or entity["confidence"] > seen[text]["confidence"]:
                        seen[text] = entity
                
                filtered_entities = list(seen.values())
                
                # Sort by confidence (descending)
                filtered_entities.sort(key=lambda x: x["confidence"], reverse=True)
                
                merged[entity_type] = filtered_entities
            
            return dict(merged)
            
        except Exception as e:
            logger.error(f"Error merging entities: {str(e)}")
            return {}
    
    def extract_all_entities(self, text: str, methods: List[str] = None) -> Dict[str, Any]:
        """Extract entities using all available methods"""
        try:
            if methods is None:
                methods = ["spacy", "transformer", "custom"]
            
            results = {
                "entities": {},
                "methods_used": methods,
                "extraction_time": datetime.now().isoformat(),
                "text_length": len(text)
            }
            
            entity_sources = []
            
            # Extract using each method
            if "spacy" in methods:
                spacy_entities = self.extract_entities_spacy(text)
                entity_sources.append(spacy_entities)
            
            if "transformer" in methods:
                transformer_entities = self.extract_entities_transformer(text)
                entity_sources.append(transformer_entities)
            
            if "custom" in methods:
                custom_entities = self.extract_custom_entities(text)
                entity_sources.append(custom_entities)
            
            # Merge all entities
            if entity_sources:
                results["entities"] = self.merge_entities(*entity_sources)
            
            # Add statistics
            results["statistics"] = self._calculate_entity_stats(results["entities"])
            
            return results
            
        except Exception as e:
            logger.error(f"Error extracting all entities: {str(e)}")
            return {
                "entities": {},
                "methods_used": methods or [],
                "extraction_time": datetime.now().isoformat(),
                "text_length": len(text),
                "error": str(e)
            }
    
    def _calculate_entity_stats(self, entities: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Calculate statistics about extracted entities"""
        try:
            stats = {
                "total_entities": 0,
                "entity_types": len(entities),
                "type_distribution": {},
                "confidence_distribution": {
                    "high": 0,  # > 0.8
                    "medium": 0,  # 0.5 - 0.8
                    "low": 0  # < 0.5
                }
            }
            
            for entity_type, entity_list in entities.items():
                count = len(entity_list)
                stats["total_entities"] += count
                stats["type_distribution"][entity_type] = count
                
                # Calculate confidence distribution
                for entity in entity_list:
                    confidence = entity.get("confidence", 0.0)
                    if confidence > 0.8:
                        stats["confidence_distribution"]["high"] += 1
                    elif confidence > 0.5:
                        stats["confidence_distribution"]["medium"] += 1
                    else:
                        stats["confidence_distribution"]["low"] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating entity stats: {str(e)}")
            return {}
    
    def get_entity_relationships(self, entities: Dict[str, List[Dict]], 
                               text: str) -> List[Dict[str, Any]]:
        """Find relationships between entities"""
        try:
            relationships = []
            
            # Simple co-occurrence based relationships
            entity_positions = []
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    entity_positions.append({
                        "type": entity_type,
                        "text": entity["text"],
                        "start": entity["start"],
                        "end": entity["end"]
                    })
            
            # Sort by position
            entity_positions.sort(key=lambda x: x["start"])
            
            # Find entities that appear close to each other
            for i in range(len(entity_positions)):
                for j in range(i + 1, len(entity_positions)):
                    entity1 = entity_positions[i]
                    entity2 = entity_positions[j]
                    
                    # Check if entities are within 100 characters of each other
                    if entity2["start"] - entity1["end"] < 100:
                        relationships.append({
                            "entity1": {
                                "type": entity1["type"],
                                "text": entity1["text"]
                            },
                            "entity2": {
                                "type": entity2["type"],
                                "text": entity2["text"]
                            },
                            "relationship_type": "co-occurrence",
                            "distance": entity2["start"] - entity1["end"]
                        })
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error finding entity relationships: {str(e)}")
            return []
    
    def extract_article_entities(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entities from a complete article"""
        try:
            title = article_data.get("title", "")
            content = article_data.get("content", "")
            
            # Combine title and content (give title more weight)
            full_text = f"{title}. {content}"
            
            # Extract entities
            result = self.extract_all_entities(full_text)
            
            # Add article metadata
            result["article_id"] = article_data.get("id", "")
            result["article_title"] = title
            result["article_source"] = article_data.get("source", "")
            
            # Find relationships
            result["relationships"] = self.get_entity_relationships(
                result["entities"], full_text
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting article entities: {str(e)}")
            return {}
    
    def process_articles_batch(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple articles for entity extraction"""
        try:
            results = []
            
            for article in articles:
                result = self.extract_article_entities(article)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing articles batch: {str(e)}")
            return []
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get NER agent statistics"""
        return {
            "spacy_model": self.spacy_model_name,
            "transformer_model": self.transformer_model_name,
            "device": self.device,
            "spacy_available": self.nlp is not None,
            "transformer_available": self.transformer_ner is not None,
            "custom_patterns": len(self.custom_patterns),
            "entity_filters": len(self.entity_filters)
        }

# Utility functions
def test_ner_agent():
    """Test function for the NER agent"""
    agent = NERAgent()
    
    test_text = """
    Apple Inc. CEO Tim Cook announced today that the company's stock (AAPL) rose 5.2% to $150.25 
    after the quarterly earnings report. The announcement was made at Apple's headquarters in 
    Cupertino, California. Cook can be reached at tcook@apple.com or (408) 996-1010.
    The company reported revenue of $81.4 billion for the quarter ending March 31, 2024.
    """
    
    print("Testing NER Agent...")
    print(f"Text length: {len(test_text)} characters")
    
    # Test entity extraction
    result = agent.extract_all_entities(test_text)
    
    print(f"\nExtracted entities using methods: {result['methods_used']}")
    print(f"Total entities found: {result['statistics']['total_entities']}")
    
    # Display entities by type
    for entity_type, entities in result["entities"].items():
        print(f"\n{entity_type}:")
        for entity in entities:
            print(f"  - {entity['text']} (confidence: {entity['confidence']:.2f}, source: {entity['source']})")
    
    # Display statistics
    print(f"\nStatistics:")
    stats = result["statistics"]
    print(f"  Entity types: {stats['entity_types']}")
    print(f"  Type distribution: {stats['type_distribution']}")
    print(f"  Confidence distribution: {stats['confidence_distribution']}")
    
    # Test article processing
    print(f"\nTesting article processing...")
    article_data = {
        "id": "test_article_1",
        "title": "Apple Reports Strong Quarterly Earnings",
        "content": test_text,
        "source": "Tech News"
    }
    
    article_result = agent.extract_article_entities(article_data)
    print(f"Article entities: {len(article_result.get('entities', {}))}")
    print(f"Relationships found: {len(article_result.get('relationships', []))}")
    
    # Show agent stats
    print(f"\nAgent statistics:")
    agent_stats = agent.get_agent_stats()
    for key, value in agent_stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_ner_agent()