"""
Courtroom AI Assistant - Main Application
Real-time transcription, context retrieval, and response generation for legal proceedings.
"""

# Load environment variables from .env file first
try:
    from dotenv import load_dotenv
    load_dotenv()  # This loads .env file automatically
except ImportError:
    pass  # python-dotenv not installed, will use system environment variables

import asyncio
import argparse
import os
import time
import signal
from pathlib import Path
from typing import Dict, List, Optional
import toml
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

# Import our modules
from audio_stream import AudioStreamer
from transcribe import WhisperTranscriber
from retriever import DocumentRetriever
from responder import LegalResponder

console = Console()

class CourtroomAssistant:
    """
    Main Courtroom AI Assistant that coordinates all components for real-time legal support.
    """
    
    def __init__(self, config_path: str = "config.toml"):
        """
        Initialize the Courtroom Assistant with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self.load_config(config_path)
        self.running = False
        
        # Initialize components
        api_key = os.getenv("OPENAI_API_KEY") or self.config["api"]["openai_api_key"]
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment or config.toml")
        
        # Audio streaming
        self.audio_streamer = AudioStreamer(
            sample_rate=self.config["audio"]["sample_rate"],
            channels=self.config["audio"]["channels"],
            chunk_duration=self.config["audio"]["chunk_duration_seconds"],
            device_index=self.config["audio"]["device_index"]
        )
        
        # Transcription
        self.transcriber = WhisperTranscriber(
            api_key=api_key,
            model=self.config["api"]["whisper_model"],
            max_retries=self.config["response"]["max_retries"]
        )
        
        # Document retrieval
        self.retriever = DocumentRetriever(
            api_key=api_key,
            embedding_model=self.config["api"]["embedding_model"]
        )
        
        # Response generation
        self.responder = LegalResponder(
            api_key=api_key,
            model=self.config["api"]["chat_model"],
            temperature=self.config["response"]["temperature"],
            max_tokens=self.config["response"]["max_tokens"]
        )
        
        # Turn detection and processing
        self.silence_threshold_ms = self.config["audio"]["silence_threshold_ms"]
        self.current_turn_text = ""
        self.last_activity_time = time.time()
        self.pending_audio_chunks = []
        
        # Statistics
        self.session_stats = {
            "turns_processed": 0,
            "responses_generated": 0,
            "total_processing_time": 0,
            "start_time": time.time()
        }
        
        console.print("[bold green]🏛️ Courtroom AI Assistant initialized[/bold green]")
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from TOML file."""
        try:
            with open(config_path, 'r') as f:
                config = toml.load(f)
            console.print(f"[green]Configuration loaded from {config_path}[/green]")
            return config
        except FileNotFoundError:
            console.print(f"[red]Config file not found: {config_path}[/red]")
            raise
        except Exception as e:
            console.print(f"[red]Failed to load config: {e}[/red]")
            raise
    
    async def initialize_index(self) -> bool:
        """Initialize document index for retrieval."""
        index_directory = self.config["paths"]["index_directory"]
        
        if not self.retriever.load_index(index_directory):
            console.print(f"[yellow]No index found in {index_directory}[/yellow]")
            console.print("[yellow]Run 'python index_builder.py' to create document index[/yellow]")
            return False
        
        self.retriever.print_stats()
        return True
    
    async def handle_turn(self, turn_text: str) -> None:
        """
        Handle a complete speaking turn with transcription, retrieval, and response.
        
        Args:
            turn_text: Complete transcribed text from the turn
        """
        if not turn_text.strip():
            return
        
        start_time = time.time()
        self.session_stats["turns_processed"] += 1
        
        console.print(f"\n[bold cyan]🎙️ Turn {self.session_stats['turns_processed']}:[/bold cyan]")
        console.print(f"[blue]'{turn_text}'[/blue]")
        
        try:
            # Detect speech patterns
            speech_patterns = self.transcriber.detect_speech_patterns(turn_text)
            console.print(f"[dim]Patterns: {speech_patterns}[/dim]")
            
            # Only generate response if speech needs one
            if speech_patterns.get("needs_response", False):
                # Retrieve relevant context
                search_results, context = await self.retriever.search_and_format(
                    turn_text,
                    top_k=self.config["retrieval"]["max_retrieved_chunks"],
                    similarity_threshold=self.config["retrieval"]["similarity_threshold"]
                )
                
                # Generate response
                response = await self.responder.generate_response(
                    turn_text,
                    context,
                    speech_patterns,
                    timeout=self.config["response"]["response_timeout_seconds"]
                )
                
                # Display response
                if response["success"]:
                    formatted_response = self.responder.format_response_for_display(response, turn_text)
                    console.print(formatted_response)
                    self.session_stats["responses_generated"] += 1
                else:
                    console.print(f"[red]❌ Response failed:[/red] {response.get('error', 'Unknown error')}")
            
            else:
                console.print("[dim]💭 No response needed[/dim]")
            
            # Update processing time
            processing_time = time.time() - start_time
            self.session_stats["total_processing_time"] += processing_time
            
            console.print(f"[dim]⏱️ Total processing time: {processing_time:.2f}s[/dim]")
            
        except Exception as e:
            console.print(f"[red]❌ Turn processing failed: {e}[/red]")
    
    def detect_turn_end(self, text_chunk: str, timestamp: float) -> bool:
        """
        Detect if a speaking turn has ended based on silence and punctuation.
        
        Args:
            text_chunk: Latest transcribed text chunk
            timestamp: Timestamp of the chunk
            
        Returns:
            True if turn has ended, False otherwise
        """
        # Update activity time if we got meaningful text
        if text_chunk.strip():
            self.last_activity_time = timestamp
            return False
        
        # Check for silence duration
        silence_duration = (timestamp - self.last_activity_time) * 1000  # Convert to ms
        
        if silence_duration > self.silence_threshold_ms:
            return True
        
        # Check for terminal punctuation in accumulated text
        if self.current_turn_text.strip().endswith(('.', '!', '?')):
            return True
        
        return False
    
    async def process_audio_stream(self) -> None:
        """Main audio processing loop with real-time transcription and turn detection."""
        console.print("[cyan]🎧 Starting audio processing...[/cyan]")
        
        try:
            self.audio_streamer.start_recording()
            
            for wav_bytes, timestamp in self.audio_streamer.get_audio_chunks():
                if not self.running:
                    break
                
                # Add chunk to pending list for parallel processing
                self.pending_audio_chunks.append((wav_bytes, timestamp))
                
                # Process chunks in parallel when we have enough or detect potential turn end
                if len(self.pending_audio_chunks) >= 2 or (time.time() - self.last_activity_time > 1.0):
                    await self.process_pending_chunks()
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
        except Exception as e:
            console.print(f"[red]Audio processing error: {e}[/red]")
        finally:
            self.audio_streamer.cleanup()
    
    async def process_pending_chunks(self) -> None:
        """Process all pending audio chunks in parallel."""
        if not self.pending_audio_chunks:
            return
        
        # Transcribe chunks in parallel
        transcription_results = await self.transcriber.transcribe_chunks_parallel(
            self.pending_audio_chunks
        )
        
        # Process each transcription result
        for result in transcription_results:
            if result["success"] and result["text"].strip():
                # Add to current turn text
                self.current_turn_text += " " + result["text"]
                
                # Check if turn has ended
                if self.detect_turn_end(result["text"], result["timestamp"]):
                    # Process the complete turn
                    await self.handle_turn(self.current_turn_text.strip())
                    
                    # Reset for next turn
                    self.current_turn_text = ""
                    self.last_activity_time = time.time()
        
        # Clear processed chunks
        self.pending_audio_chunks = []
    
    async def run_dry_run(self, wav_file: str) -> None:
        """
        Run in dry-run mode with a pre-recorded WAV file for testing.
        
        Args:
            wav_file: Path to WAV file for testing
        """
        console.print(f"[cyan]🧪 Running dry-run with {wav_file}[/cyan]")
        
        if not Path(wav_file).exists():
            console.print(f"[red]WAV file not found: {wav_file}[/red]")
            return
        
        try:
            # Read WAV file
            with open(wav_file, 'rb') as f:
                wav_bytes = f.read()
            
            # Transcribe the file
            result = await self.transcriber.transcribe_chunk(wav_bytes, time.time())
            
            if result["success"] and result["text"].strip():
                await self.handle_turn(result["text"])
            else:
                console.print("[yellow]No transcription result from WAV file[/yellow]")
                
        except Exception as e:
            console.print(f"[red]Dry-run failed: {e}[/red]")
    
    async def run(self, dry_run_wav: Optional[str] = None) -> None:
        """
        Run the main application loop.
        
        Args:
            dry_run_wav: Optional WAV file for dry-run mode
        """
        self.running = True
        
        # Initialize document index
        if not await self.initialize_index():
            console.print("[red]Cannot run without document index[/red]")
            return
        
        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            console.print("\n[yellow]Shutting down gracefully...[/yellow]")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            if dry_run_wav:
                # Run in dry-run mode
                await self.run_dry_run(dry_run_wav)
            else:
                # Run in live mode
                console.print("\n[bold green]🚀 Courtroom AI Assistant is now active![/bold green]")
                console.print("[dim]Listening for speech... Press Ctrl+C to stop.[/dim]")
                
                await self.process_audio_stream()
        
        finally:
            self.print_session_stats()
    
    def print_session_stats(self) -> None:
        """Print session statistics."""
        duration = time.time() - self.session_stats["start_time"]
        
        console.print("\n[bold]📊 Session Statistics:[/bold]")
        console.print(f"  Duration: {duration:.1f}s")
        console.print(f"  Turns processed: {self.session_stats['turns_processed']}")
        console.print(f"  Responses generated: {self.session_stats['responses_generated']}")
        console.print(f"  Total processing time: {self.session_stats['total_processing_time']:.1f}s")
        
        if self.session_stats["turns_processed"] > 0:
            avg_time = self.session_stats["total_processing_time"] / self.session_stats["turns_processed"]
            console.print(f"  Avg processing time per turn: {avg_time:.2f}s")
        
        # Print component statistics
        self.transcriber.print_statistics()
        self.responder.print_statistics()


async def main():
    """Main entry point for the Courtroom AI Assistant."""
    parser = argparse.ArgumentParser(
        description="Courtroom AI Assistant - Real-time legal transcription and response",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Run in live mode
  python main.py --dry-run test.wav       # Test with WAV file
  python main.py --config custom.toml     # Use custom config
  python main.py --list-devices           # List audio devices
        """
    )
    
    parser.add_argument("--config", default="config.toml", 
                       help="Configuration file path (default: config.toml)")
    parser.add_argument("--dry-run", metavar="WAV_FILE",
                       help="Run in dry-run mode with specified WAV file")
    parser.add_argument("--list-devices", action="store_true",
                       help="List available audio input devices and exit")
    
    args = parser.parse_args()
    
    # Handle device listing
    if args.list_devices:
        from audio_stream import list_audio_devices
        list_audio_devices()
        return
    
    try:
        # Create and run the assistant
        assistant = CourtroomAssistant(args.config)
        await assistant.run(dry_run_wav=args.dry_run)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Application error: {e}[/red]")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 