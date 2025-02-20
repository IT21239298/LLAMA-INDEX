import os
import asyncio
import json
from typing import Dict, Any, List

from dotenv import load_dotenv

# Import agent classes
from base_agent import BaseAgent
from database_agent import DatabaseAgent
from web_agent import WebAgent

class MultiAgentCoordinator:
    """
    Coordinates multiple agents to process queries.
    Implements a strict priority-based query resolution strategy.
    """
    def __init__(self, mongo_uri: str):
        """
        Initialize MultiAgentCoordinator.
        
        Args:
            mongo_uri: MongoDB connection string
        """
        # Initialize agents
        self.database_agent = DatabaseAgent(mongo_uri)
        self.web_agent = WebAgent()
        
        # Define query patterns for specific database searches
        self.query_patterns = {
            'recent contacts': {
                'collection': 'contacts',
                'limit': 10
            },
            'recent customers': {
                'collection': 'customers',
                'limit': 10
            },
            'recent messages': {
                'collection': 'messages',
                'limit': 10
            }
        }
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process query with strict database-first approach.
        
        Args:
            query: User's input query
        
        Returns:
            Processed query results
        """
        # Normalize query
        normalized_query = query.lower().strip()
        
        try:
            # 1. First, try specific database search for known query patterns
            for pattern, config in self.query_patterns.items():
                if pattern in normalized_query:
                    try:
                        # Directly get recent documents from specified collection
                        results_json = self.database_agent.get_recent_documents(
                            config['collection'], 
                            limit=config.get('limit', 5)
                        )
                        
                        # Parse results
                        results = json.loads(results_json)
                        
                        # Check if results are valid
                        if not isinstance(results, dict) or 'error' not in results:
                            return {
                                'agent': 'DatabaseAgent',
                                'results': {
                                    'source': 'database',
                                    'results': {config['collection']: results},
                                    'confidence': 1.0,
                                    'query': query
                                },
                                'confidence': 1.0
                            }
                    except Exception as e:
                        print(f"Specific query error: {e}")
            
            # 2. If no specific pattern match, try general database search
            db_result = await self.database_agent.process_query(query)
            
            # Check if database search was successful
            if db_result['results'] and len(db_result['results']) > 0:
                return {
                    'agent': 'DatabaseAgent',
                    'results': db_result,
                    'confidence': db_result['confidence']
                }
            
            # 3. If database search fails, try web search
            web_result = await self.web_agent.process_query(query)
            
            # Check if web search was successful
            if web_result['results'] and web_result['results'].get('answer'):
                return {
                    'agent': 'WebAgent',
                    'results': web_result,
                    'confidence': web_result['confidence']
                }
            
            # 4. If both searches fail
            return {
                'agent': None,
                'results': None,
                'confidence': 0.0,
                'error': 'No relevant information found'
            }
        
        except Exception as e:
            return {
                'agent': None,
                'results': None,
                'confidence': 0.0,
                'error': str(e)
            }
    
    async def interactive_chat(self):
        """
        Interactive chat interface for multi-agent system.
        """
        print("\nMulti-Agent Intelligent Assistant")
        print("Type 'exit' to quit")
        
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                # Exit condition
                if user_input.lower() == 'exit':
                    break
                
                # Process query
                result = await self.process_query(user_input)
                
                # Display results
                if result['agent']:
                    print(f"\nResponse from {result['agent']} (Confidence: {result.get('confidence', 0.0):.2f}):")
                    
                    # Handle different result types
                    if result['agent'] == 'DatabaseAgent':
                        for collection, docs in result['results']['results'].items():
                            print(f"\n{collection.capitalize()} Results:")
                            for doc in docs:
                                print(json.dumps(doc, indent=2))
                    
                    elif result['agent'] == 'WebAgent':
                        print("\nWeb Search Result:")
                        print(result['results']['results'].get('answer', 'No detailed answer'))
                        
                        sources = result['results']['results'].get('sources', [])
                        if sources:
                            print("\nSources:")
                            for source in sources:
                                print(f"- {source}")
                else:
                    print("\nSorry, I couldn't find an answer to your query.")
            
            except Exception as e:
                print(f"\nError processing query: {e}")

def main():
    """
    Main entry point for the multi-agent system.
    """
    # Load environment variables
    load_dotenv()
    
    # Get MongoDB URI from environment
    mongo_uri = os.getenv("MONGODB_URI")
    
    if not mongo_uri:
        print("Error: MONGODB_URI not found in environment variables")
        return
    
    # Create and run multi-agent coordinator
    coordinator = MultiAgentCoordinator(mongo_uri)
    asyncio.run(coordinator.interactive_chat())

if __name__ == "__main__":
    main()