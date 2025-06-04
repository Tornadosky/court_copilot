"""
Document retrieval module for similarity search using FAISS index.
Finds relevant document chunks based on query embeddings.
"""

import asyncio
import numpy as np
from typing import List, Dict, Tuple, Optional
import faiss
from openai import AsyncOpenAI
from rich.console import Console

console = Console()

class DocumentRetriever:
    """
    Document retriever that performs similarity search on indexed documents.
    Uses OpenAI embeddings and FAISS for fast vector search.
    """
    
    def __init__(self, api_key: str, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize the document retriever.
        
        Args:
            api_key: OpenAI API key for generating query embeddings
            embedding_model: OpenAI embedding model to use (must match indexing model)
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.embedding_model = embedding_model
        
        # Loaded index components
        self.index = None           # FAISS index
        self.chunks = []           # Chunk metadata
        self.documents = []        # Document metadata
        
        console.print(f"[green]Document retriever initialized:[/green] {embedding_model}")
    
    def load_index(self, index_directory: str) -> bool:
        """
        Load FAISS index and metadata from disk.
        
        Args:
            index_directory: Directory containing saved index files
            
        Returns:
            True if loaded successfully, False otherwise
        """
        from pathlib import Path
        import json
        
        index_path = Path(index_directory)
        faiss_file = index_path / "faiss.index"
        metadata_file = index_path / "metadata.json"
        
        if not (faiss_file.exists() and metadata_file.exists()):
            console.print(f"[red]Index files not found in {index_directory}[/red]")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(faiss_file))
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            self.documents = metadata["documents"]
            self.chunks = metadata["chunks"]
            
            console.print(f"[green]Index loaded successfully:[/green] {len(self.chunks)} chunks from {len(self.documents)} documents")
            
            return True
            
        except Exception as e:
            console.print(f"[red]Failed to load index: {e}[/red]")
            return False
    
    async def get_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding as numpy array
        """
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=[query]
            )
            
            embedding = np.array(response.data[0].embedding)
            return embedding
            
        except Exception as e:
            console.print(f"[red]Failed to generate query embedding: {e}[/red]")
            raise
    
    async def search(self, query: str, top_k: int = 5, similarity_threshold: float = 0.7) -> List[Dict]:
        """
        Search for relevant document chunks based on a query.
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score (0-1, higher = more similar)
            
        Returns:
            List of relevant chunks with similarity scores and metadata
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call load_index() first.")
        
        if not query.strip():
            return []
        
        # Generate query embedding
        query_embedding = await self.get_query_embedding(query)
        
        # Perform similarity search using FAISS
        # Note: FAISS returns L2 distances, we convert to similarity scores
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32), 
            min(top_k * 2, self.index.ntotal)  # Get more results for filtering
        )
        
        # Convert L2 distances to similarity scores (0-1 scale)
        # Lower L2 distance = higher similarity
        max_distance = np.max(distances[0]) if len(distances[0]) > 0 else 1.0
        similarities = 1 - (distances[0] / (max_distance + 1e-6))  # Avoid division by zero
        
        # Filter and format results
        results = []
        for i, (chunk_idx, similarity) in enumerate(zip(indices[0], similarities)):
            if chunk_idx == -1:  # FAISS returns -1 for invalid indices
                continue
            
            if similarity < similarity_threshold:
                continue  # Skip results below threshold
            
            # Get chunk and document metadata
            chunk = self.chunks[chunk_idx]
            document = self.documents[chunk["document_id"]]
            
            result = {
                "chunk_text": chunk["text"],
                "similarity_score": float(similarity),
                "chunk_id": chunk["chunk_id"],
                "document_name": document["file_name"],
                "document_path": document["file_path"],
                "document_type": document["file_type"],
                "token_count": chunk["token_count"],
                "rank": i + 1
            }
            
            results.append(result)
        
        # Sort by similarity score (highest first) and limit to top_k
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        results = results[:top_k]
        
        # Log search results
        if results:
            console.print(f"[blue]🔍 Found {len(results)} relevant chunks for query:[/blue] '{query[:50]}{'...' if len(query) > 50 else ''}'")
            for i, result in enumerate(results[:3]):  # Show top 3
                console.print(f"  {i+1}: {result['document_name']} (score: {result['similarity_score']:.3f})")
        else:
            console.print(f"[yellow]No relevant chunks found for query:[/yellow] '{query}'")
        
        return results
    
    def format_context(self, search_results: List[Dict], max_tokens: int = 1500) -> str:
        """
        Format search results into a context string for the LLM.
        
        Args:
            search_results: List of search results from search()
            max_tokens: Maximum number of tokens in formatted context
            
        Returns:
            Formatted context string
        """
        if not search_results:
            return "No relevant documents found."
        
        context_parts = []
        current_tokens = 0
        
        for result in search_results:
            # Create a formatted chunk with metadata
            chunk_header = f"\n[Document: {result['document_name']} | Score: {result['similarity_score']:.3f}]"
            chunk_text = result['chunk_text']
            
            # Estimate tokens (rough approximation)
            chunk_tokens = len(chunk_text.split()) * 1.3  # Rough token estimation
            
            if current_tokens + chunk_tokens > max_tokens:
                break  # Stop if we would exceed token limit
            
            context_parts.append(f"{chunk_header}\n{chunk_text}")
            current_tokens += chunk_tokens
        
        formatted_context = "\n\n".join(context_parts)
        
        return formatted_context
    
    async def search_and_format(self, query: str, top_k: int = 5, 
                              similarity_threshold: float = 0.7, 
                              max_context_tokens: int = 1500) -> Tuple[List[Dict], str]:
        """
        Search for relevant chunks and format them for LLM context.
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score
            max_context_tokens: Maximum tokens in formatted context
            
        Returns:
            Tuple of (search_results, formatted_context)
        """
        search_results = await self.search(query, top_k, similarity_threshold)
        formatted_context = self.format_context(search_results, max_context_tokens)
        
        return search_results, formatted_context
    
    def get_document_stats(self) -> Dict:
        """Get statistics about the loaded document index."""
        if not self.documents or not self.chunks:
            return {"error": "No index loaded"}
        
        # Calculate document type distribution
        doc_types = {}
        for doc in self.documents:
            doc_type = doc["file_type"]
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        # Calculate chunk statistics
        chunk_tokens = [chunk["token_count"] for chunk in self.chunks]
        avg_chunk_tokens = sum(chunk_tokens) / len(chunk_tokens) if chunk_tokens else 0
        
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "document_types": doc_types,
            "average_chunk_tokens": avg_chunk_tokens,
            "total_tokens": sum(doc["token_count"] for doc in self.documents),
            "index_size": self.index.ntotal if self.index else 0
        }
    
    def print_stats(self) -> None:
        """Print document index statistics to console."""
        stats = self.get_document_stats()
        
        if "error" in stats:
            console.print(f"[red]{stats['error']}[/red]")
            return
        
        console.print("\n[bold]Document Index Statistics:[/bold]")
        console.print(f"  Documents: {stats['total_documents']}")
        console.print(f"  Chunks: {stats['total_chunks']}")
        console.print(f"  Total tokens: {stats['total_tokens']:,}")
        console.print(f"  Avg chunk size: {stats['average_chunk_tokens']:.1f} tokens")
        console.print(f"  Document types: {dict(stats['document_types'])}")


async def test_retriever():
    """Test function for the document retriever."""
    import os
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]OPENAI_API_KEY environment variable not set[/red]")
        return
    
    retriever = DocumentRetriever(api_key)
    
    # Try to load an existing index
    if not retriever.load_index("./index"):
        console.print("[yellow]No index found. Run index_builder.py first.[/yellow]")
        return
    
    retriever.print_stats()
    
    # Test queries
    test_queries = [
        "What is the statute of limitations?",
        "burden of proof",
        "contract formation requirements",
        "evidence admissibility",
        "motion to dismiss"
    ]
    
    console.print("\n[bold]Testing retrieval with sample queries:[/bold]")
    
    for query in test_queries:
        console.print(f"\n[cyan]Query:[/cyan] {query}")
        
        try:
            results, context = await retriever.search_and_format(query, top_k=3)
            
            if results:
                console.print(f"[green]Found {len(results)} results[/green]")
                console.print(f"[dim]Context preview: {context[:200]}...[/dim]")
            else:
                console.print("[yellow]No results found[/yellow]")
                
        except Exception as e:
            console.print(f"[red]Search failed: {e}[/red]")


if __name__ == "__main__":
    asyncio.run(test_retriever()) 