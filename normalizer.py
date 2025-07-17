# processing/normalizer.py

from langdetect import detect
from newspaper import Article

def normalize_article(article):
    url = article.get("link", "")
    try:
        news_article = Article(url)
        news_article.download()
        news_article.parse()
    except Exception as e:
        return None  # Skip if download/parsing fails

    cleaned_article = {
        "title": news_article.title,
        "link": url,
        "text": news_article.text.strip(),
        "published": article.get("published", ""),
        "language": detect(news_article.text) if news_article.text else "unknown"
    }
    return cleaned_article
