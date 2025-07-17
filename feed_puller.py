# ingestion/feed_puller.py
import feedparser

def fetch_articles(feed_url):
    feed = feedparser.parse(feed_url)
    articles = []

    for entry in feed.entries:
        articles.append({
            'title': entry.title,
            'link': entry.link,
            'summary': entry.summary,
            'published': entry.published
        })

    return articles
