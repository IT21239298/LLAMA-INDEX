import os
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional

from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings

from base_agent import BaseAgent

class WebAgent(BaseAgent):
    """
    Agent responsible for web-based searches and content retrieval.
    """
    def __init__(self, base_url: str = "https://econsulate.net/"):
        """
        Initialize WebAgent.
        
        Args:
            base_url: Base URL for web content
        """
        super().__init__(name="WebAgent", priority=2)
        
        # Web search configuration
        self.base_url = base_url
        self.index = None
        self.initialization_started = False
        self.documents: List[Document] = []
        
        # Initialize OpenAI models
        self.llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
        self.embed_model = OpenAIEmbedding()
        
        # Configure settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        print("Web Agent initialized. Starting content loading...")

    async def fetch_page(self, url: str, session: aiohttp.ClientSession) -> str:
        """Fetch page content with retry logic."""
        max_retries = 3 # Maximum number of attempts to fetch the page
        for attempt in range(max_retries):
            try:
                async with session.get(url, timeout=10) as response:
                    # Try to get the page with 10 second timeout
                    if response.status == 200:
                        return await response.text()
                    await asyncio.sleep(1)  # Wait before retry
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to fetch {url}: {str(e)}")
                await asyncio.sleep(1)
        return ""

    def process_page(self, html: str, url: str) -> Optional[Document]:
        """Process page content into a Document."""
        try:
            # BeautifulSoup is a Python library used for pulling data out of HTML and XML files
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style']):
                element.decompose()

            # Get main content
            # Try to find main content in this order: <main>, <article>, or div with class='content'
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
            title = soup.title.string if soup.title else ''
            
            if main_content:
                text = main_content.get_text(strip=True, separator=' ')
            else:
                text = soup.get_text(strip=True, separator=' ')

            # Clean and format text
            text = ' '.join(text.split())  # Normalize whitespace
            
            # Return Document object with text and metadata
            return Document(text=text, metadata={'url': url, 'title': title})
        except Exception as e:
            print(f"Error processing page {url}: {str(e)}")
            return None

    async def crawl_pages(self) -> List[Document]:
        """Crawl website pages with progress updates."""
        documents = []  # List to store processed documents
        visited = set()  # Set to track visited URLs
        to_visit = [self.base_url] # List of URLs to visit, starting with base URL
        pages_processed = 0  # Counter for processed pages
        total_pages = min(len(to_visit), 50)  # Limit to 50 pages

        print("\nStarting website crawl...")
        # Create session for HTTP requests
        async with aiohttp.ClientSession() as session:
            while to_visit and pages_processed < total_pages:
                url = to_visit.pop(0) # Get next URL to process
                # Skip if already visited
                if url in visited:
                    continue

                print(f"Processing page {pages_processed + 1}/{total_pages}...")
                # Fetch page content
                html = await self.fetch_page(url, session)
                # Process the HTML content
                if html:
                    doc = self.process_page(html, url)
                    if doc:
                        documents.append(doc) # Add to documents list
                        visited.add(url)   # Mark as visited
                        pages_processed += 1 # Increment counter

                        # Extract more links
                        soup = BeautifulSoup(html, 'html.parser')
                        for a in soup.find_all('a', href=True):
                            next_url = urljoin(url, a['href'])
                            if (next_url.startswith(self.base_url) and 
                                next_url not in visited and 
                                next_url not in to_visit):
                                to_visit.append(next_url)

                await asyncio.sleep(0.1)  # Prevent overwhelming the server

        return documents

    async def initialize_index(self):
        """Initialize the search index."""
        try:
            print("Creating search index...")
            if not self.documents:
                self.documents = await self.crawl_pages()
            
            if self.documents:
                self.index = VectorStoreIndex.from_documents(self.documents)
                print(f"\nInitialization complete! Indexed {len(self.documents)} pages.")
                return True
            else:
                print("\nNo documents were successfully processed.")
                return False
                
        except Exception as e:
            print(f"\nError during initialization: {str(e)}")
            return False

    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Perform web-based search for the query.
        
        Args:
            query: User's input query
        
        Returns:
            Dictionary with search results
        """
        # Ensure web index is initialized
        if not self.initialization_started:
            self.initialization_started = True
            success = await self.initialize_index()
            if not success:
                return {
                    'source': 'web',
                    'results': {},
                    'confidence': 0.0,
                    'query': query,
                    'error': "I'm having trouble accessing the website. Please try again later."
                }

        if not self.index:
            return {
                'source': 'web',
                'results': {},
                'confidence': 0.0,
                'query': query,
                'error': "Still initializing... Please wait a moment and try again."
            }

        try:
            # Create query engine
            query_engine = self.index.as_query_engine(similarity_top_k=3)
            
            # Perform query
            response = query_engine.query(query)
            
            # Prepare results
            results = {
                'answer': str(response),
                'sources': []
            }
            
            # Extract source URLs if available
            if hasattr(response, 'source_nodes'):
                results['sources'] = [
                    node.metadata.get('url', 'Unknown Source') 
                    for node in response.source_nodes
                ]
            
            # Calculate confidence based on response quality
            confidence = 0.5 if results['answer'] else 0.0
            
            return {
                'source': 'web',
                'results': results,
                'confidence': confidence,
                'query': query,
                'error': None
            }
        
        except Exception as e:
            return {
                'source': 'web',
                'results': {},
                'confidence': 0.0,
                'query': query,
                'error': str(e)
            }