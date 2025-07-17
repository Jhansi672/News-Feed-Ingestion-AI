# vector_store/indexer.py
import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone
import uuid
import json
from loguru import logger
from dataclasses import asdict
import os

class VectorStoreIndexer:
    """Vector store indexer using ChromaDB for news articles"""
    
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "news_articles"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Ensure directory exists
            os.makedirs(self.db_path, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except ValueError:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "News articles vector store"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB client: {str(e)}")
            raise
    
    def _prepare_metadata(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare metadata for ChromaDB (only JSON serializable types)"""
        metadata = {}
        
        # Basic article information
        if 'title' in article_data:
            metadata['title'] = str(article_data['title'])
        if 'author' in article_data:
            metadata['author'] = str(article_data['author']) if article_data['author'] else ""
        if 'source' in article_data:
            metadata['source'] = str(article_data['source']) if article_data['source'] else ""
        if 'url' in article_data:
            metadata['url'] = str(article_data['url'])
        
        # Dates
        if 'published' in article_data and article_data['published']:
            if isinstance(article_data['published'], datetime):
                metadata['published'] = article_data['published'].isoformat()
            else:
                metadata['published'] = str(article_data['published'])
        
        # Content metadata
        if 'word_count' in article_data:
            metadata['word_count'] = int(article_data['word_count'])
        if 'reading_time' in article_data:
            metadata['reading_time'] = int(article_data['reading_time'])
        if 'language' in article_data:
            metadata['language'] = str(article_data['language'])
        
        # Categories and tags
        if 'category' in article_data:
            metadata['category'] = str(article_data['category'])
        if 'tags' in article_data and article_data['tags']:
            metadata['tags'] = json.dumps(article_data['tags'])
        
        # Sentiment and classification
        if 'sentiment' in article_data:
            metadata['sentiment'] = str(article_data['sentiment'])
        if 'sentiment_score' in article_data:
            metadata['sentiment_score'] = float(article_data['sentiment_score'])
        
        # Processing metadata
        metadata['indexed_at'] = datetime.now(timezone.utc).isoformat()
        
        return metadata
    
    def add_article(self, article_id: str, embedding: np.ndarray, 
                   content: str, metadata: Dict[str, Any]) -> bool:
        """Add a single article to the vector store"""
        try:
            if not self.collection:
                logger.error("Collection not initialized")
                return False
            
            # Prepare metadata
            prepared_metadata = self._prepare_metadata(metadata)
            
            # Add to collection
            self.collection.add(
                embeddings=[embedding.tolist()],
                documents=[content],
                metadatas=[prepared_metadata],
                ids=[article_id]
            )
            
            logger.debug(f"Added article {article_id} to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding article to vector store: {str(e)}")
            return False
    
    def add_articles_batch(self, articles_data: List[Dict[str, Any]]) -> int:
        """Add multiple articles to the vector store in batch"""
        try:
            if not self.collection or not articles_data:
                return 0
            
            embeddings = []
            documents = []
            metadatas = []
            ids = []
            
            for article_data in articles_data:
                # Validate required fields
                if not all(key in article_data for key in ['id', 'embedding', 'content']):
                    logger.warning(f"Skipping article with missing required fields")
                    continue
                
                embeddings.append(article_data['embedding'].tolist())
                documents.append(article_data['content'])
                metadatas.append(self._prepare_metadata(article_data))
                ids.append(article_data['id'])
            
            if not embeddings:
                logger.warning("No valid articles to add")
                return 0
            
            # Add batch to collection
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(embeddings)} articles to vector store")
            return len(embeddings)
            
        except Exception as e:
            logger.error(f"Error adding articles batch: {str(e)}")
            return 0
    
    def search_similar(self, query_embedding: np.ndarray, n_results: int = 10,
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar articles using vector similarity"""
        try:
            if not self.collection:
                logger.error("Collection not initialized")
                return []
            
            # Prepare where clause for filtering
            where_clause = {}
            if filters:
                for key, value in filters.items():
                    if isinstance(value, (str, int, float)):
                        where_clause[key] = value
                    elif isinstance(value, list) and len(value) == 2:
                        # Range filter [min, max]
                        where_clause[key] = {"$gte": value[0], "$lte": value[1]}
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    result = {
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching similar articles: {str(e)}")
            return []
    
    def search_by_text(self, query_text: str, n_results: int = 10,
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search articles by text query (requires embedding generation)"""
        try:
            if not self.collection:
                logger.error("Collection not initialized")
                return []
            
            # This would typically require the embedding generator
            # For now, we'll use ChromaDB's built-in text search if available
            where_clause = {}
            if filters:
                for key, value in filters.items():
                    if isinstance(value, (str, int, float)):
                        where_clause[key] = value
            
            # Use ChromaDB's text search capability
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    result = {
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'similarity': 1 - results['distances'][0][i]
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching by text: {str(e)}")
            return []
    
    def get_article_by_id(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Get article by ID"""
        try:
            if not self.collection:
                return None
            
            results = self.collection.get(
                ids=[article_id],
                include=["documents", "metadatas"]
            )
            
            if results['ids'] and results['ids'][0]:
                return {
                    'id': results['ids'][0],
                    'content': results['documents'][0],
                    'metadata': results['metadatas'][0]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting article by ID: {str(e)}")
            return None
    
    def update_article_metadata(self, article_id: str, metadata: Dict[str, Any]) -> bool:
        """Update article metadata"""
        try:
            if not self.collection:
                return False
            
            prepared_metadata = self._prepare_metadata(metadata)
            
            self.collection.update(
                ids=[article_id],
                metadatas=[prepared_metadata]
            )
            
            logger.debug(f"Updated metadata for article {article_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating article metadata: {str(e)}")
            return False
    
    def delete_article(self, article_id: str) -> bool:
        """Delete article from vector store"""
        try:
            if not self.collection:
                return False
            
            self.collection.delete(ids=[article_id])
            logger.debug(f"Deleted article {article_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting article: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            if not self.collection:
                return {}
            
            count = self.collection.count()
            
            return {
                'collection_name': self.collection_name,
                'total_articles': count,
                'db_path': self.db_path
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}
    
    def filter_articles(self, filters: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """Filter articles by metadata"""
        try:
            if not self.collection:
                return []
            
            where_clause = {}
            for key, value in filters.items():
                if isinstance(value, (str, int, float)):
                    where_clause[key] = value
                elif isinstance(value, dict):
                    where_clause[key] = value
                elif isinstance(value, list) and len(value) == 2:
                    where_clause[key] = {"$gte": value[0], "$lte": value[1]}
            
            results = self.collection.get(
                where=where_clause,
                limit=limit,
                include=["documents", "metadatas"]
            )
            
            formatted_results = []
            if results['ids']:
                for i in range(len(results['ids'])):
                    result = {
                        'id': results['ids'][i],
                        'content': results['documents'][i],
                        'metadata': results['metadatas'][i]
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error filtering articles: {str(e)}")
            return []
    
    def get_articles_by_date_range(self, start_date: str, end_date: str, 
                                  limit: int = 100) -> List[Dict[str, Any]]:
        """Get articles within date range"""
        filters = {
            "published": {
                "$gte": start_date,
                "$lte": end_date
            }
        }
        return self.filter_articles(filters, limit)
    
    def get_articles_by_source(self, source: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get articles from specific source"""
        filters = {"source": source}
        return self.filter_articles(filters, limit)
    
    def get_articles_by_category(self, category: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get articles by category"""
        filters = {"category": category}
        return self.filter_articles(filters, limit)
    
    def reset_collection(self):
        """Reset the collection (delete all data)"""
        try:
            if self.client:
                self.client.delete_collection(name=self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "News articles vector store"}
                )
                logger.info(f"Reset collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
    
    def backup_collection(self, backup_path: str):
        """Backup collection data"""
        try:
            # This is a simple backup - in production, you'd want more sophisticated backup
            all_articles = self.collection.get(include=["documents", "metadatas"])
            
            backup_data = {
                'collection_name': self.collection_name,
                'articles': all_articles,
                'backup_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            logger.info(f"Collection backed up to {backup_path}")
            
        except Exception as e:
            logger.error(f"Error backing up collection: {str(e)}")

# Utility functions
def test_vector_store():
    """Test function for the vector store indexer"""
    indexer = VectorStoreIndexer(db_path="./test_chroma_db")
    
    # Test data
    test_articles = [
        {
            'id': 'test_1',
            'embedding': np.random.rand(384),  # Typical embedding size
            'content': 'This is a test article about technology.',
            'title': 'Test Article 1',
            'source': 'Test Source',
            'category': 'Technology',
            'published': datetime.now(timezone.utc)
        },
        {
            'id': 'test_2',
            'embedding': np.random.rand(384),
            'content': 'This is another test article about sports.',
            'title': 'Test Article 2',
            'source': 'Test Source',
            'category': 'Sports',
            'published': datetime.now(timezone.utc)
        }
    ]
    
    # Test adding articles
    print("Testing batch article addition...")
    added_count = indexer.add_articles_batch(test_articles)
    print(f"Added {added_count} articles")
    
    # Test search
    print("\nTesting similarity search...")
    query_embedding = np.random.rand(384)
    results = indexer.search_similar(query_embedding, n_results=2)
    print(f"Found {len(results)} similar articles")
    
    # Test filtering
    print("\nTesting filtering...")
    tech_articles = indexer.get_articles_by_category('Technology')
    print(f"Found {len(tech_articles)} technology articles")
    
    # Test stats
    print("\nCollection stats:")
    stats = indexer.get_collection_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    indexer.reset_collection()
    print("\nTest completed and collection reset")

if __name__ == "__main__":
    test_vector_store()