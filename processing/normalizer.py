# processing/normalizer.py
import re
import html
from typing import Optional, Dict, Any
from bs4 import BeautifulSoup
from loguru import logger
from urllib.parse import urljoin, urlparse
import unicodedata

class ContentNormalizer:
    """Normalizes and cleans article content"""
    
    def __init__(self, min_length: int = 100, max_length: int = 10000):
        self.min_length = min_length
        self.max_length = max_length
        
        # Compile regex patterns for efficiency
        self.patterns = {
            'whitespace': re.compile(r'\s+'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'phone': re.compile(r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'),
            'extra_punctuation': re.compile(r'[.]{3,}|[!]{2,}|[?]{2,}'),
            'social_mentions': re.compile(r'@\w+|#\w+'),
            'advertisement': re.compile(r'\b(advertisement|sponsored|ad|promo)\b', re.IGNORECASE),
            'subscribe': re.compile(r'\b(subscribe|newsletter|follow us)\b', re.IGNORECASE)
        }
    
    def _clean_html(self, content: str) -> str:
        """Remove HTML tags and decode entities"""
        if not content:
            return ""
        
        # Decode HTML entities
        content = html.unescape(content)
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
        
        # Remove common ad containers
        for ad_container in soup.find_all(class_=re.compile(r'ad|advertisement|sponsored|promo', re.I)):
            ad_container.decompose()
        
        # Get text content
        text = soup.get_text()
        
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters"""
        # Normalize unicode to NFC form
        text = unicodedata.normalize('NFC', text)
        
        # Replace common unicode quotation marks
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('–', '-').replace('—', '-')
        text = text.replace('…', '...')
        
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean and normalize whitespace"""
        # Replace multiple whitespaces with single space
        text = self.patterns['whitespace'].sub(' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove empty lines and excessive line breaks
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:  # Only keep non-empty lines
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _remove_boilerplate(self, text: str) -> str:
        """Remove common boilerplate text"""
        # Remove advertisement mentions
        text = self.patterns['advertisement'].sub('', text)
        
        # Remove subscription prompts
        text = self.patterns['subscribe'].sub('', text)
        
        # Remove social media mentions (optional)
        # text = self.patterns['social_mentions'].sub('', text)
        
        # Remove excessive punctuation
        text = self.patterns['extra_punctuation'].sub('...', text)
        
        return text
    
    def _extract_main_content(self, text: str) -> str:
        """Extract main content, removing headers/footers"""
        lines = text.split('\n')
        
        # Find the main content by looking for the longest continuous block
        content_blocks = []
        current_block = []
        
        for line in lines:
            line = line.strip()
            if len(line) > 50:  # Likely content line
                current_block.append(line)
            else:
                if current_block:
                    content_blocks.append('\n'.join(current_block))
                    current_block = []
        
        if current_block:
            content_blocks.append('\n'.join(current_block))
        
        # Return the longest block as main content
        if content_blocks:
            main_content = max(content_blocks, key=len)
            return main_content
        
        return text
    
    def _validate_content(self, content: str) -> bool:
        """Validate if content meets quality criteria"""
        if not content:
            return False
        
        # Check length
        if len(content) < self.min_length or len(content) > self.max_length:
            return False
        
        # Check if it's mostly punctuation or numbers
        alpha_ratio = sum(c.isalpha() for c in content) / len(content)
        if alpha_ratio < 0.5:
            return False
        
        # Check for minimum sentence structure
        sentences = content.split('.')
        if len(sentences) < 2:
            return False
        
        return True
    
    def normalize_content(self, content: str) -> Optional[str]:
        """Main method to normalize article content"""
        try:
            if not content:
                return None
            
            logger.debug(f"Normalizing content of length: {len(content)}")
            
            # Step 1: Clean HTML
            content = self._clean_html(content)
            
            # Step 2: Normalize unicode
            content = self._normalize_unicode(content)
            
            # Step 3: Clean whitespace
            content = self._clean_whitespace(content)
            
            # Step 4: Remove boilerplate
            content = self._remove_boilerplate(content)
            
            # Step 5: Extract main content
            content = self._extract_main_content(content)
            
            # Step 6: Final whitespace cleanup
            content = self._clean_whitespace(content)
            
            # Step 7: Validate content
            if not self._validate_content(content):
                logger.warning("Content failed validation")
                return None
            
            logger.debug(f"Normalized content length: {len(content)}")
            return content
            
        except Exception as e:
            logger.error(f"Error normalizing content: {str(e)}")
            return None
    
    def normalize_title(self, title: str) -> Optional[str]:
        """Normalize article title"""
        try:
            if not title:
                return None
            
            # Clean HTML
            title = self._clean_html(title)
            
            # Normalize unicode
            title = self._normalize_unicode(title)
            
            # Clean whitespace
            title = title.strip()
            
            # Remove excessive punctuation
            title = self.patterns['extra_punctuation'].sub('...', title)
            
            # Validate title
            if len(title) < 10 or len(title) > 200:
                return None
            
            return title
            
        except Exception as e:
            logger.error(f"Error normalizing title: {str(e)}")
            return None
    
    def normalize_summary(self, summary: str, max_length: int = 300) -> Optional[str]:
        """Normalize article summary"""
        try:
            if not summary:
                return None
            
            # Clean HTML
            summary = self._clean_html(summary)
            
            # Normalize unicode
            summary = self._normalize_unicode(summary)
            
            # Clean whitespace
            summary = self._clean_whitespace(summary)
            
            # Truncate if too long
            if len(summary) > max_length:
                # Try to truncate at sentence boundary
                sentences = summary.split('.')
                truncated = ""
                for sentence in sentences:
                    if len(truncated + sentence + '.') <= max_length:
                        truncated += sentence + '.'
                    else:
                        break
                
                if truncated:
                    summary = truncated
                else:
                    summary = summary[:max_length] + '...'
            
            # Validate summary
            if len(summary) < 20:
                return None
            
            return summary
            
        except Exception as e:
            logger.error(f"Error normalizing summary: {str(e)}")
            return None
    
    def get_content_stats(self, content: str) -> Dict[str, Any]:
        """Get statistics about the content"""
        if not content:
            return {}
        
        words = content.split()
        sentences = content.split('.')
        paragraphs = content.split('\n\n')
        
        return {
            'character_count': len(content),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'avg_words_per_sentence': len(words) / len(sentences) if sentences else 0,
            'avg_chars_per_word': len(content) / len(words) if words else 0
        }

# Utility functions
def test_normalizer():
    """Test function for the content normalizer"""
    normalizer = ContentNormalizer()
    
    test_content = """
    <html>
    <head><title>Test</title></head>
    <body>
        <div class="advertisement">This is an ad</div>
        <p>This is a test article with <strong>HTML tags</strong> and &quot;HTML entities&quot;.</p>
        <p>It has multiple paragraphs    with    extra    whitespace.</p>
        <script>alert('remove me');</script>
        <p>Subscribe to our newsletter for more content!</p>
        <p>This is the main content that should be preserved and normalized properly.</p>
    </body>
    </html>
    """
    
    normalized = normalizer.normalize_content(test_content)
    print("Original content:")
    print(test_content)
    print("\nNormalized content:")
    print(normalized)
    print("\nContent stats:")
    print(normalizer.get_content_stats(normalized))

if __name__ == "__main__":
    test_normalizer()