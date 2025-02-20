import os
import asyncio
import json
from typing import Dict, Any, Optional

# Core libraries
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime

# LlamaIndex and OpenAI
from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.settings import Settings

# Web scraping
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin

class MultiAgentCoordinator:
    def __init__(self):
        """
        Initialize multi-agent system with integrated database and web search capabilities.
        """
        # Load environment variables
        load_dotenv()

        # Initialize MongoDB
        mongo_uri = os.getenv("MONGODB_URI")
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client.tn_txbglk

        # Initialize OpenAI models
        self.llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
        self.embed_model = OpenAIEmbedding()

        # Configure global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        # Web search configuration
        self.base_url = "https://econsulate.net/"
        self.web_index = None
        self.web_documents = []

        # Agent configuration
        self.agent_config = {
            'database': {
                'priority': 1,
                'confidence_threshold': 0.8,
                'collections': [
                    'contacts', 'customers', 'conversations', 
                    'messages', 'cases'
                ]
            },
            'web': {
                'priority': 2,
                'confidence_threshold': 0.5,
                'max_pages': 50
            }
        }

        # Initialize web search index
        asyncio.run(self.initialize_web_index())

    async def initialize_web_index(self):
        """
        Asynchronously initialize web search index.
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Crawl web pages
                self.web_documents = await self._crawl_pages(session)
            
            # Create vector index if documents found
            if self.web_documents:
                self.web_index = VectorStoreIndex.from_documents(self.web_documents)
                print(f"Web index initialized with {len(self.web_documents)} documents")
        except Exception as e:
            print(f"Web index initialization error: {e}")

    async def _crawl_pages(self, session: aiohttp.ClientSession) -> list:
        """
        Crawl web pages and convert to documents.
        
        Args:
            session: Async HTTP client session
        
        Returns:
            List of Document objects
        """
        documents = []
        visited = set()
        to_visit = [self.base_url]
        
        while to_visit and len(documents) < self.agent_config['web']['max_pages']:
            url = to_visit.pop(0)
            
            if url in visited:
                continue
            
            try:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        html = await response.text()
                        doc = self._process_web_page(html, url)
                        
                        if doc:
                            documents.append(doc)
                            visited.add(url)
                        
                        # Extract additional links
                        soup = BeautifulSoup(html, 'html.parser')
                        for link in soup.find_all('a', href=True):
                            next_url = urljoin(url, link['href'])
                            if (next_url.startswith(self.base_url) and 
                                next_url not in visited and 
                                next_url not in to_visit):
                                to_visit.append(next_url)
            except Exception as e:
                print(f"Error crawling {url}: {e}")
        
        return documents

    def _process_web_page(self, html: str, url: str) -> Optional[Document]:
        """
        Process HTML content into a Document.
        
        Args:
            html: HTML content
            url: Source URL
        
        Returns:
            Document object or None
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style']):
                element.decompose()
            
            # Extract main content
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
            title = soup.title.string if soup.title else url
            
            text = main_content.get_text(strip=True, separator=' ') if main_content else soup.get_text(strip=True, separator=' ')
            text = ' '.join(text.split())  # Normalize whitespace
            
            return Document(text=text, metadata={'url': url, 'title': title})
        except Exception as e:
            print(f"Error processing web page {url}: {e}")
            return None

    async def search_database(self, query: str) -> Dict[str, Any]:
        """
        Search across database collections.
        
        Args:
            query: Search query
        
        Returns:
            Dictionary with search results
        """
        results = {}
        
        # Search configured collections
        for collection_name in self.agent_config['database']['collections']:
            try:
                collection = self.db[collection_name]
                
                # Flexible search across multiple fields
                search_query = {
                    '$or': [
                        {'firstName': {'$regex': query, '$options': 'i'}},
                        {'lastName': {'$regex': query, '$options': 'i'}},
                        {'email': {'$regex': query, '$options': 'i'}},
                        {'name': {'$regex': query, '$options': 'i'}},
                        {'title': {'$regex': query, '$options': 'i'}},
                        {'description': {'$regex': query, '$options': 'i'}}
                    ]
                }
                
                # Fetch results
                collection_results = list(collection.find(search_query).limit(5))
                
                # Format results (handle MongoDB-specific types)
                formatted_results = []
                for doc in collection_results:
                    formatted_doc = {}
                    for key, value in doc.items():
                        if isinstance(value, ObjectId):
                            formatted_doc[key] = str(value)
                        elif isinstance(value, datetime):
                            formatted_doc[key] = value.isoformat()
                        else:
                            formatted_doc[key] = value
                    formatted_results.append(formatted_doc)
                
                # Store results if found
                if formatted_results:
                    results[collection_name] = formatted_results
            
            except Exception as e:
                print(f"Error searching {collection_name}: {e}")
        
        return results

    async def search_web(self, query: str) -> Dict[str, Any]:
        """
        Search web index for relevant information.
        
        Args:
            query: Search query
        
        Returns:
            Dictionary with web search results
        """
        # Ensure web index is initialized
        if not self.web_index:
            await self.initialize_web_index()
        
        try:
            # Create query engine
            query_engine = self.web_index.as_query_engine(similarity_top_k=3)
            
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
            
            return results
        
        except Exception as e:
            print(f"Web search error: {e}")
            return {}

    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Coordinate search across database and web sources.
        
        Args:
            query: User's search query
        
        Returns:
            Comprehensive search results
        """
        # Search database first
        database_results = await self.search_database(query)
        
        # If database search yields results, return them
        if database_results:
            return {
                'source': 'database',
                'confidence': self.agent_config['database']['confidence_threshold'],
                'results': database_results
            }
        
        # Fallback to web search
        web_results = await self.search_web(query)
        
        # If web search yields results, return them
        if web_results:
            return {
                'source': 'web',
                'confidence': self.agent_config['web']['confidence_threshold'],
                'results': web_results
            }
        
        # No results found
        return {
            'source': 'none',
            'confidence': 0,
            'results': {},
            'message': 'No relevant information found'
        }

    async def interactive_chat(self):
        """
        Interactive chat interface for multi-agent system.
        """
        print("\nMulti-Agent Search Assistant")
        print("Type 'exit' to quit")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == 'exit':
                    break
                
                # Process query
                results = await self.process_query(user_input)
                
                # Display results
                print(f"\nSource: {results['source']} (Confidence: {results['confidence']})")
                
                if results['source'] == 'database':
                    for collection, docs in results['results'].items():
                        print(f"\n{collection.capitalize()} Results:")
                        for doc in docs:
                            print(json.dumps(doc, indent=2))
                
                elif results['source'] == 'web':
                    print("\nWeb Search Result:")
                    print(results['results'].get('answer', 'No detailed answer'))
                    
                    sources = results['results'].get('sources', [])
                    if sources:
                        print("\nSources:")
                        for source in sources:
                            print(f"- {source}")
                
                else:
                    print(results.get('message', 'No results found'))
            
            except Exception as e:
                print(f"\nError processing query: {e}")

def main():
    """Main entry point for the multi-agent coordinator."""
    coordinator = MultiAgentCoordinator()
    asyncio.run(coordinator.interactive_chat())

if __name__ == "__main__":
    main()