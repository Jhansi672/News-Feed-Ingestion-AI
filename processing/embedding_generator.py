# processing/embedding_generator.py
import numpy as np
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
from loguru import logger
import asyncio
import torch
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pickle

class EmbeddingGenerator:
    """Generates embeddings for news articles using sentence transformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 batch_size: int = 32, max_length: int = 512):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Model loaded successfully on device: {self.device}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _generate_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for embedding generation"""
        if not text:
            return ""
        
        # Truncate if too long
        if len(text) > self.max_length * 4:  # Rough character to token ratio
            text = text[:self.max_length * 4]
        
        # Clean up text
        text = text.strip()
        text = ' '.join(text.split())  # Normalize whitespace
        
        return text
    
    def generate_single_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for a single text"""
        try:
            if not text or not self.model:
                return None
            
            # Check cache first
            cache_key = self._generate_cache_key(text)
            if cache_key in self.embedding_cache:
                return self.embedding_cache[cache_key]
            
            # Preprocess text
            processed_text = self._preprocess_text(text)
            if not processed_text:
                return None
            
            # Generate embedding
            embedding = self.model.encode(processed_text, convert_to_numpy=True)
            
            # Cache the result
            self.embedding_cache[cache_key] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None
    
    def generate_batch_embeddings(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Generate embeddings for a batch of texts"""
        try:
            if not texts or not self.model:
                return [None] * len(texts)
            
            # Preprocess texts and check cache
            processed_texts = []
            results = [None] * len(texts)
            indices_to_process = []
            
            for i, text in enumerate(texts):
                if not text:
                    continue
                
                cache_key = self._generate_cache_key(text)
                if cache_key in self.embedding_cache:
                    results[i] = self.embedding_cache[cache_key]
                else:
                    processed_text = self._preprocess_text(text)
                    if processed_text:
                        processed_texts.append(processed_text)
                        indices_to_process.append(i)
            
            # Generate embeddings for uncached texts
            if processed_texts:
                embeddings = self.model.encode(
                    processed_texts, 
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=len(processed_texts) > 10
                )
                
                # Store results and cache
                for idx, embedding in zip(indices_to_process, embeddings):
                    results[idx] = embedding
                    cache_key = self._generate_cache_key(texts[idx])
                    self.embedding_cache[cache_key] = embedding
            
            return results
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            return [None] * len(texts)
    
    async def generate_embeddings_async(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Generate embeddings asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.generate_batch_embeddings, texts)
    
    def generate_article_embeddings(self, article_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Generate multiple embeddings for different parts of an article"""
        embeddings = {}
        
        try:
            # Generate embeddings for different components
            components = {
                'title': article_data.get('title', ''),
                'content': article_data.get('content', ''),
                'summary': article_data.get('summary', ''),
                'combined': f"{article_data.get('title', '')} {article_data.get('summary', '')}"
            }
            
            # Generate embeddings for each component
            for component_name, text in components.items():
                if text:
                    embedding = self.generate_single_embedding(text)
                    if embedding is not None:
                        embeddings[component_name] = embedding
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating article embeddings: {str(e)}")
            return {}
    
    def generate_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Generate embedding for search query"""
        return self.generate_single_embedding(query)
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def find_similar_articles(self, query_embedding: np.ndarray, 
                            article_embeddings: List[np.ndarray], 
                            threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find articles similar to query embedding"""
        try:
            similarities = []
            
            for i, article_embedding in enumerate(article_embeddings):
                if article_embedding is not None:
                    similarity = self.calculate_similarity(query_embedding, article_embedding)
                    if similarity >= threshold:
                        similarities.append({
                            'index': i,
                            'similarity': similarity
                        })
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            return similarities
            
        except Exception as e:
            logger.error(f"Error finding similar articles: {str(e)}")
            return []
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding generator"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'cache_size': len(self.embedding_cache),
            'embedding_dimension': self.model.get_sentence_embedding_dimension() if self.model else None
        }
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self.embedding_cache.clear()
        logger.info("Embedding cache cleared")
    
    def save_cache(self, filepath: str):
        """Save embedding cache to file"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.info(f"Embedding cache saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
    
    def load_cache(self, filepath: str):
        """Load embedding cache from file"""
        try:
            with open(filepath, 'rb') as f:
                self.embedding_cache = pickle.load(f)
            logger.info(f"Embedding cache loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

class MultiModalEmbeddingGenerator(EmbeddingGenerator):
    """Extended embedding generator with support for different embedding types"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.specialized_models = {}
    
    def load_specialized_model(self, model_type: str, model_name: str):
        """Load specialized model for specific embedding type"""
        try:
            self.specialized_models[model_type] = SentenceTransformer(model_name, device=self.device)
            logger.info(f"Loaded specialized model for {model_type}: {model_name}")
        except Exception as e:
            logger.error(f"Error loading specialized model: {str(e)}")
    
    def generate_specialized_embedding(self, text: str, model_type: str = "default") -> Optional[np.ndarray]:
        """Generate embedding using specialized model"""
        if model_type == "default" or model_type not in self.specialized_models:
            return self.generate_single_embedding(text)
        
        try:
            model = self.specialized_models[model_type]
            processed_text = self._preprocess_text(text)
            if not processed_text:
                return None
            
            embedding = model.encode(processed_text, convert_to_numpy=True)
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating specialized embedding: {str(e)}")
            return None
    
    def generate_hierarchical_embeddings(self, article_data: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
        """Generate hierarchical embeddings at different levels"""
        embeddings = {
            'sentence_level': {},
            'paragraph_level': {},
            'document_level': {}
        }
        
        try:
            content = article_data.get('content', '')
            if not content:
                return embeddings
            
            # Document level
            embeddings['document_level']['full'] = self.generate_single_embedding(content)
            
            # Paragraph level
            paragraphs = content.split('\n\n')
            for i, paragraph in enumerate(paragraphs):
                if paragraph.strip():
                    embedding = self.generate_single_embedding(paragraph)
                    if embedding is not None:
                        embeddings['paragraph_level'][f'paragraph_{i}'] = embedding
            
            # Sentence level (first few sentences for efficiency)
            sentences = content.split('.')[:10]  # Limit to first 10 sentences
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    embedding = self.generate_single_embedding(sentence)
                    if embedding is not None:
                        embeddings['sentence_level'][f'sentence_{i}'] = embedding
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating hierarchical embeddings: {str(e)}")
            return embeddings

# Utility functions
def test_embedding_generator():
    """Test function for the embedding generator"""
    generator = EmbeddingGenerator()
    
    test_texts = [
        "Apple Inc. announced new products today.",
        "The stock market closed higher yesterday.",
        "Scientists discovered a new planet in distant galaxy.",
        "Local weather forecast shows rain tomorrow."
    ]
    
    print("Testing single embedding generation...")
    embedding = generator.generate_single_embedding(test_texts[0])
    print(f"Generated embedding shape: {embedding.shape if embedding is not None else 'None'}")
    
    print("\nTesting batch embedding generation...")
    embeddings = generator.generate_batch_embeddings(test_texts)
    print(f"Generated {len([e for e in embeddings if e is not None])} embeddings out of {len(test_texts)}")
    
    print("\nTesting similarity calculation...")
    if embeddings[0] is not None and embeddings[1] is not None:
        similarity = generator.calculate_similarity(embeddings[0], embeddings[1])
        print(f"Similarity between first two texts: {similarity:.4f}")
    
    print("\nEmbedding generator stats:")
    stats = generator.get_embedding_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

async def test_async_embedding():
    """Test async embedding generation"""
    generator = EmbeddingGenerator()
    
    test_texts = [
        "Breaking news: Major earthquake hits the region.",
        "Technology stocks surge after positive earnings.",
        "Climate change summit begins in Paris.",
        "New medical breakthrough announced by researchers."
    ]
    
    print("Testing async embedding generation...")
    embeddings = await generator.generate_embeddings_async(test_texts)
    print(f"Generated {len([e for e in embeddings if e is not None])} embeddings asynchronously")

if __name__ == "__main__":
    test_embedding_generator()
    # asyncio.run(test_async_embedding())