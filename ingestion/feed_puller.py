# ingestion/feed_puller.py
import asyncio
import aiohttp
import feedparser
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from loguru import logger
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import hashlib
import time

@dataclass
class Article:
    """Data class for news articles"""
    id: str
    title: str
    content: str
    summary: str
    url: str
    published: datetime
    author: Optional[str] = None
    source: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class FeedPuller:
    """RSS feed puller for news ingestion"""
    
    def __init__(self, feeds: List[str], max_articles_per_feed: int = 50):
        self.feeds = feeds
        self.max_articles_per_feed = max_articles_per_feed
        self.session = None
        self.processed_articles = set()  # Track processed article IDs
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'AI News Feed Service/1.0'
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _generate_article_id(self, url: str, title: str) -> str:
        """Generate unique article ID from URL and title"""
        content = f"{url}:{title}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _parse_published_date(self, published_parsed) -> datetime:
        """Parse published date from feedparser"""
        if published_parsed:
            return datetime(*published_parsed[:6], tzinfo=timezone.utc)
        return datetime.now(timezone.utc)
    
    async def _fetch_feed(self, feed_url: str) -> Optional[Dict[str, Any]]:
        """Fetch and parse a single RSS feed"""
        try:
            logger.info(f"Fetching feed: {feed_url}")
            
            async with self.session.get(feed_url) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    if feed.bozo:
                        logger.warning(f"Feed parsing warning for {feed_url}: {feed.bozo_exception}")
                    
                    return feed
                else:
                    logger.error(f"Failed to fetch {feed_url}: HTTP {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching feed {feed_url}: {str(e)}")
            return None
    
    def _extract_article_content(self, entry: Dict[str, Any]) -> str:
        """Extract article content from feed entry"""
        content = ""
        
        # Try different content fields
        if hasattr(entry, 'content') and entry.content:
            content = entry.content[0].value
        elif hasattr(entry, 'summary') and entry.summary:
            content = entry.summary
        elif hasattr(entry, 'description') and entry.description:
            content = entry.description
        
        return content
    
    def _parse_article(self, entry: Dict[str, Any], source: str) -> Optional[Article]:
        """Parse a single article from feed entry"""
        try:
            title = entry.get('title', '').strip()
            url = entry.get('link', '').strip()
            
            if not title or not url:
                return None
            
            article_id = self._generate_article_id(url, title)
            
            # Skip if already processed
            if article_id in self.processed_articles:
                return None
            
            content = self._extract_article_content(entry)
            summary = entry.get('summary', content[:200] + '...' if len(content) > 200 else content)
            
            published = self._parse_published_date(entry.get('published_parsed'))
            author = entry.get('author', '').strip() or None
            
            # Extract tags
            tags = []
            if hasattr(entry, 'tags') and entry.tags:
                tags = [tag.term for tag in entry.tags if hasattr(tag, 'term')]
            
            article = Article(
                id=article_id,
                title=title,
                content=content,
                summary=summary,
                url=url,
                published=published,
                author=author,
                source=source,
                tags=tags
            )
            
            self.processed_articles.add(article_id)
            return article
            
        except Exception as e:
            logger.error(f"Error parsing article: {str(e)}")
            return None
    
    async def pull_articles(self) -> List[Article]:
        """Pull articles from all configured feeds"""
        all_articles = []
        
        # Fetch all feeds concurrently
        tasks = [self._fetch_feed(feed_url) for feed_url in self.feeds]
        feeds = await asyncio.gather(*tasks, return_exceptions=True)
        
        for feed_url, feed in zip(self.feeds, feeds):
            if isinstance(feed, Exception):
                logger.error(f"Exception for feed {feed_url}: {feed}")
                continue
                
            if not feed or not hasattr(feed, 'entries'):
                logger.warning(f"No entries found for feed: {feed_url}")
                continue
            
            source = feed.feed.get('title', urlparse(feed_url).netloc)
            logger.info(f"Processing {len(feed.entries)} entries from {source}")
            
            # Process entries (limit to max_articles_per_feed)
            entries_to_process = feed.entries[:self.max_articles_per_feed]
            
            for entry in entries_to_process:
                article = self._parse_article(entry, source)
                if article:
                    all_articles.append(article)
        
        logger.info(f"Successfully pulled {len(all_articles)} new articles")
        return all_articles
    
    async def pull_single_feed(self, feed_url: str) -> List[Article]:
        """Pull articles from a single feed"""
        feed = await self._fetch_feed(feed_url)
        if not feed or not hasattr(feed, 'entries'):
            return []
        
        articles = []
        source = feed.feed.get('title', urlparse(feed_url).netloc)
        
        for entry in feed.entries[:self.max_articles_per_feed]:
            article = self._parse_article(entry, source)
            if article:
                articles.append(article)
        
        return articles
    
    def clear_processed_cache(self):
        """Clear the processed articles cache"""
        self.processed_articles.clear()
        logger.info("Cleared processed articles cache")

# Utility functions
async def test_feed_puller():
    """Test function for the feed puller"""
    test_feeds = [
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://rss.cnn.com/rss/edition.rss"
    ]
    
    async with FeedPuller(test_feeds, max_articles_per_feed=5) as puller:
        articles = await puller.pull_articles()
        
        print(f"Pulled {len(articles)} articles:")
        for article in articles[:3]:  # Show first 3
            print(f"- {article.title} ({article.source})")
            print(f"  URL: {article.url}")
            print(f"  Published: {article.published}")
            print()

if __name__ == "__main__":
    asyncio.run(test_feed_puller())