# AI News Feed Ingestion & Processing Service

A comprehensive AI-powered news feed ingestion and processing service that automatically collects, processes, and analyzes news articles from multiple sources.

## Features

- **RSS Feed Ingestion**: Automatically pulls news from multiple RSS feeds
- **Content Processing**: Normalizes and extracts metadata from articles
- **AI Analysis**: Sentiment analysis, NER, and content classification
- **Vector Storage**: Stores embeddings for semantic search
- **RAG Capabilities**: Retrieval-Augmented Generation for news insights
- **REST API**: FastAPI-based API for accessing processed news

## Architecture

```
├── main.py                 # Entry point
├── config/
│   └── settings.py         # Configuration management
├── ingestion/
│   └── feed_puller.py      # RSS feed ingestion
├── processing/
│   ├── normalizer.py       # Content normalization
│   ├── metadata_extractor.py  # Metadata extraction
│   └── embedding_generator.py # Vector embeddings
├── vector_store/
│   └── indexer.py          # Vector database operations
├── agents/
│   ├── classifier.py       # Content classification
│   ├── ner_agent.py        # Named Entity Recognition
│   ├── sentiment_agent.py  # Sentiment analysis
│   ├── rag_agent.py        # RAG implementation
│   └── orchestrator.py     # Main orchestration
└── api/
    └── server.py           # FastAPI server
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download required models:
   ```bash
   python -m spacy download en_core_web_sm
   ```
4. Set up environment variables:
   ```bash
   export OPENAI_API_KEY=your_openai_api_key
   export CHROMA_DB_PATH=./chroma_db
   ```

## Usage

### Running the Service

```bash
python main.py
```

### API Endpoints

- `GET /health` - Health check
- `GET /articles` - Retrieve processed articles
- `POST /articles/search` - Semantic search
- `GET /articles/{article_id}` - Get specific article
- `POST /analyze` - Analyze text content

### Configuration

Modify `config/settings.py` to customize:
- RSS feed sources
- Processing intervals
- Model configurations
- API settings

## Components

### Ingestion
- Pulls RSS feeds from configured sources
- Handles rate limiting and error recovery
- Deduplicates content

### Processing
- Normalizes article content
- Extracts metadata (title, author, date, etc.)
- Generates embeddings for semantic search

### AI Agents
- **Classifier**: Categorizes articles by topic
- **NER Agent**: Extracts named entities
- **Sentiment Agent**: Analyzes sentiment
- **RAG Agent**: Provides contextual responses

### Vector Store
- Stores article embeddings in ChromaDB
- Enables semantic search capabilities
- Maintains metadata indexes

## License

MIT License