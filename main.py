# main.py
import asyncio
import uvicorn
from fastapi import FastAPI
from loguru import logger
from api.server import app as api_app
from agents.orchestrator import NewsOrchestrator
from config.settings import Settings

def main():
    """Main entry point for the AI News Feed Ingestion & Processing Service"""
    settings = Settings()
    
    # Initialize the orchestrator
    orchestrator = NewsOrchestrator()
    
    # Start the background processing
    async def start_background_tasks():
        logger.info("Starting background news processing tasks...")
        await orchestrator.start_processing()
    
    # Run background tasks
    asyncio.create_task(start_background_tasks())
    
    # Start the API server
    logger.info(f"Starting API server on {settings.host}:{settings.port}")
    uvicorn.run(
        api_app,
        host=settings.host,
        port=settings.port,
        log_level="info"
    )

if __name__ == "__main__":
    main()
