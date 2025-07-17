from ingestion.feed_puller import fetch_articles
from processing.normalizer import normalize_article
import json

articles = fetch_articles("https://feeds.bbci.co.uk/news/world/rss.xml")

print(f"\n✅ Pulled {len(articles)} articles:\n")

normalized_articles = []
for article in articles:
    print(f"DEBUG ➤ Article:\n{article}\n")
    normalized = normalize_article(article)
    if normalized:
        normalized_articles.append(normalized)
        print(f"📰 {normalized['title']} — {normalized['language']}")
    else:
        print("❌ Normalization failed.\n")

with open("output/normalized_articles.json", "w", encoding="utf-8") as f:
    json.dump(normalized_articles, f, ensure_ascii=False, indent=2)

print(f"\n✅ Saved {len(normalized_articles)} normalized articles to output/normalized_articles.json")
