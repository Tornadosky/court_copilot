"""
Transcription module for converting audio to text using OpenAI Whisper API.
Handles parallel processing of audio chunks for minimal latency.
"""

import asyncio
import io
import time
from typing import Dict, Optional, List, Tuple
from openai import AsyncOpenAI
from rich.console import Console

console = Console()

class WhisperTranscriber:
    """
    Async transcriber that converts audio chunks to text using OpenAI Whisper API.
    Supports parallel processing for reduced latency.
    """
    
    def __init__(self, api_key: str, model: str = "whisper-1", max_retries: int = 3):
        """
        Initialize the Whisper transcriber.
        
        Args:
            api_key: OpenAI API key
            model: Whisper model to use ("whisper-1" is fastest)
            max_retries: Maximum number of retries for failed API calls
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        
        # Track API usage and performance
        self.total_requests = 0
        self.failed_requests = 0
        self.total_audio_seconds = 0
        self.total_transcription_time = 0
        
        console.print(f"[green]Whisper transcriber initialized:[/green] {model} model, {max_retries} max retries")
    
    async def transcribe_chunk(self, audio_bytes: bytes, timestamp: float) -> Dict:
        """
        Transcribe a single audio chunk to text.
        
        Args:
            audio_bytes: WAV format audio data
            timestamp: Timestamp when audio was captured
            
        Returns:
            Dictionary with transcription result and metadata
        """
        start_time = time.time()
        
        for attempt in range(self.max_retries + 1):
            try:
                # Create file-like object from bytes for API upload
                audio_file = io.BytesIO(audio_bytes)
                audio_file.name = "audio.wav"  # Required by OpenAI API
                
                # Call Whisper API
                response = await self.client.audio.transcriptions.create(
                    file=audio_file,
                    model=self.model,
                    response_format="text",
                    language="en"  # Assume English for courtroom usage
                )
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Update statistics
                self.total_requests += 1
                self.total_transcription_time += processing_time
                
                # Estimate audio duration (rough approximation based on file size)
                estimated_duration = len(audio_bytes) / (16000 * 2)  # 16kHz, 2 bytes per sample
                self.total_audio_seconds += estimated_duration
                
                result = {
                    "text": response.strip(),
                    "timestamp": timestamp,
                    "processing_time": processing_time,
                    "audio_duration": estimated_duration,
                    "attempt": attempt + 1,
                    "success": True
                }
                
                if result["text"]:  # Only log non-empty transcriptions
                    console.print(f"[blue]📝 Transcribed:[/blue] '{result['text'][:50]}{'...' if len(result['text']) > 50 else ''}' ({processing_time:.2f}s)")
                
                return result
                
            except Exception as e:
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    console.print(f"[yellow]Transcription attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...[/yellow]")
                    await asyncio.sleep(wait_time)
                else:
                    console.print(f"[red]Transcription failed after {self.max_retries + 1} attempts: {e}[/red]")
                    self.failed_requests += 1
                    
                    return {
                        "text": "",
                        "timestamp": timestamp,
                        "processing_time": time.time() - start_time,
                        "audio_duration": 0,
                        "attempt": attempt + 1,
                        "success": False,
                        "error": str(e)
                    }
    
    async def transcribe_chunks_parallel(self, audio_chunks: List[Tuple[bytes, float]]) -> List[Dict]:
        """
        Transcribe multiple audio chunks in parallel for improved throughput.
        
        Args:
            audio_chunks: List of (audio_bytes, timestamp) tuples
            
        Returns:
            List of transcription results in order
        """
        if not audio_chunks:
            return []
        
        start_time = time.time()
        console.print(f"[cyan]🔄 Transcribing {len(audio_chunks)} chunks in parallel...[/cyan]")
        
        # Create tasks for parallel processing
        tasks = [
            self.transcribe_chunk(audio_bytes, timestamp)
            for audio_bytes, timestamp in audio_chunks
        ]
        
        # Execute all transcriptions in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that occurred
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                console.print(f"[red]Chunk {i} transcription failed: {result}[/red]")
                processed_results.append({
                    "text": "",
                    "timestamp": audio_chunks[i][1],
                    "processing_time": 0,
                    "audio_duration": 0,
                    "attempt": 1,
                    "success": False,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        total_time = time.time() - start_time
        successful_chunks = sum(1 for r in processed_results if r["success"])
        
        console.print(f"[green]✅ Parallel transcription complete:[/green] {successful_chunks}/{len(audio_chunks)} successful in {total_time:.2f}s")
        
        return processed_results
    
    def detect_speech_patterns(self, text: str) -> Dict[str, bool]:
        """
        Detect if transcribed text contains questions, claims, or citation requests.
        Simple pattern matching for courtroom context.
        
        Args:
            text: Transcribed text to analyze
            
        Returns:
            Dictionary with detected pattern types
        """
        text_lower = text.lower().strip()
        
        # Question indicators
        is_question = (
            text.strip().endswith("?") or
            any(text_lower.startswith(q) for q in [
                "what", "when", "where", "why", "how", "who", "which",
                "is", "are", "was", "were", "do", "does", "did", "can", "could",
                "would", "will", "should", "may", "might"
            ])
        )
        
        # Citation request indicators
        is_citation_request = any(phrase in text_lower for phrase in [
            "cite", "citation", "case law", "precedent", "statute", "rule",
            "authority", "reference", "source", "legal basis", "what case",
            "which case", "find the", "look up", "research"
        ])
        
        # Claim indicators (assertions that might need counter-arguments)
        is_claim = any(phrase in text_lower for phrase in [
            "i submit", "i argue", "i contend", "i maintain", "the fact is",
            "it is clear", "obviously", "certainly", "without doubt",
            "the evidence shows", "as established", "the record reflects"
        ]) and not is_question
        
        # Objection indicators
        is_objection = any(phrase in text_lower for phrase in [
            "objection", "i object", "sustained", "overruled", "foundation",
            "relevance", "hearsay", "leading", "argumentative", "speculation"
        ])
        
        return {
            "is_question": is_question,
            "is_citation_request": is_citation_request,
            "is_claim": is_claim,
            "is_objection": is_objection,
            "needs_response": is_question or is_citation_request or is_claim
        }
    
    def get_statistics(self) -> Dict:
        """Get transcription performance statistics."""
        avg_processing_time = (
            self.total_transcription_time / self.total_requests
            if self.total_requests > 0 else 0
        )
        
        success_rate = (
            (self.total_requests - self.failed_requests) / self.total_requests * 100
            if self.total_requests > 0 else 0
        )
        
        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate_percent": success_rate,
            "total_audio_seconds": self.total_audio_seconds,
            "total_transcription_time": self.total_transcription_time,
            "average_processing_time": avg_processing_time,
            "real_time_factor": (
                self.total_transcription_time / self.total_audio_seconds
                if self.total_audio_seconds > 0 else 0
            )
        }
    
    def print_statistics(self) -> None:
        """Print transcription statistics to console."""
        stats = self.get_statistics()
        
        console.print("\n[bold]Transcription Statistics:[/bold]")
        console.print(f"  Total requests: {stats['total_requests']}")
        console.print(f"  Success rate: {stats['success_rate_percent']:.1f}%")
        console.print(f"  Audio processed: {stats['total_audio_seconds']:.1f}s")
        console.print(f"  Avg processing time: {stats['average_processing_time']:.2f}s")
        console.print(f"  Real-time factor: {stats['real_time_factor']:.2f}x")


async def test_transcriber():
    """Test function for the transcriber (requires API key)."""
    import os
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]OPENAI_API_KEY environment variable not set[/red]")
        return
    
    transcriber = WhisperTranscriber(api_key)
    
    # Test speech pattern detection
    test_texts = [
        "What is the legal precedent for this case?",
        "I object to this line of questioning.",
        "The defendant clearly violated the statute.",
        "Can you cite the relevant case law?",
        "Good morning, Your Honor."
    ]
    
    console.print("[bold]Testing speech pattern detection:[/bold]")
    for text in test_texts:
        patterns = transcriber.detect_speech_patterns(text)
        console.print(f"  '{text}' -> {patterns}")
    
    transcriber.print_statistics()


if __name__ == "__main__":
    asyncio.run(test_transcriber()) 