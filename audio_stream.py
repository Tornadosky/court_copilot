"""
Audio streaming module for real-time microphone capture.
Handles non-blocking audio capture and maintains a queue of audio chunks.
"""

import asyncio
import queue
import threading
import time
import wave
import io
from typing import Optional, Generator, Tuple
import numpy as np
import sounddevice as sd
from rich.console import Console

console = Console()

class AudioStreamer:
    """
    Real-time audio streamer that captures audio in chunks and queues them for processing.
    Designed for minimal latency and non-blocking operation.
    """
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1, 
                 chunk_duration: float = 3.0, device_index: int = -1):
        """
        Initialize the audio streamer.
        
        Args:
            sample_rate: Audio sample rate in Hz (16kHz for Whisper)
            channels: Number of audio channels (1 for mono)
            chunk_duration: Duration of each audio chunk in seconds
            device_index: Audio device index (-1 for default)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.device_index = device_index
        
        # Calculate chunk size in frames
        self.chunk_frames = int(sample_rate * chunk_duration)
        
        # Audio buffer and queue
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.stream = None
        
        # Silence detection parameters
        self.silence_threshold = 0.01  # RMS threshold for silence detection
        self.silence_duration = 0.8  # Seconds of silence to trigger turn end
        
        console.print(f"[green]Audio streamer initialized:[/green] {sample_rate}Hz, {channels}ch, {chunk_duration}s chunks")
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """
        Callback function for sounddevice stream.
        Called automatically when audio data is available.
        """
        if status:
            console.print(f"[yellow]Audio callback status:[/yellow] {status}")
        
        # Add timestamp and copy audio data to queue
        if self.is_recording:
            audio_chunk = indata.copy()
            timestamp = time.time()
            try:
                self.audio_queue.put_nowait((audio_chunk, timestamp))
            except queue.Full:
                console.print("[red]Audio queue full, dropping chunk[/red]")
    
    def start_recording(self) -> None:
        """Start recording audio from the microphone."""
        if self.is_recording:
            console.print("[yellow]Already recording[/yellow]")
            return
        
        try:
            # Initialize audio stream
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                blocksize=self.chunk_frames,
                device=self.device_index if self.device_index >= 0 else None,
                callback=self._audio_callback
            )
            
            self.stream.start()
            self.is_recording = True
            console.print("[green]🎤 Recording started[/green]")
            
        except Exception as e:
            console.print(f"[red]Failed to start recording: {e}[/red]")
            raise
    
    def stop_recording(self) -> None:
        """Stop recording audio."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        console.print("[yellow]🛑 Recording stopped[/yellow]")
    
    def get_audio_chunks(self) -> Generator[Tuple[bytes, float], None, None]:
        """
        Generator that yields audio chunks as WAV bytes.
        
        Yields:
            Tuple of (wav_bytes, timestamp)
        """
        while self.is_recording or not self.audio_queue.empty():
            try:
                # Get audio chunk from queue (non-blocking)
                audio_data, timestamp = self.audio_queue.get_nowait()
                
                # Convert numpy array to WAV bytes
                wav_bytes = self._numpy_to_wav_bytes(audio_data)
                
                yield wav_bytes, timestamp
                
            except queue.Empty:
                # No audio available, yield control
                time.sleep(0.01)  # Small sleep to prevent busy waiting
    
    def _numpy_to_wav_bytes(self, audio_data: np.ndarray) -> bytes:
        """
        Convert numpy audio array to WAV format bytes.
        
        Args:
            audio_data: Audio data as numpy array (float32, -1.0 to 1.0)
            
        Returns:
            WAV format bytes ready for API upload
        """
        # Convert float32 to int16 for WAV format
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # 2 bytes for int16
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        wav_buffer.seek(0)
        return wav_buffer.read()
    
    def detect_silence(self, audio_data: np.ndarray) -> bool:
        """
        Detect if audio chunk contains silence.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            True if audio is considered silent, False otherwise
        """
        # Calculate RMS (Root Mean Square) of the audio
        rms = np.sqrt(np.mean(audio_data ** 2))
        return rms < self.silence_threshold
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.stop_recording()
        
        # Clear any remaining audio data from memory
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        console.print("[green]Audio streamer cleaned up[/green]")


def list_audio_devices() -> None:
    """List available audio input devices for debugging."""
    console.print("[bold]Available Audio Input Devices:[/bold]")
    devices = sd.query_devices()
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:  # Only show input devices
            console.print(f"  {i}: {device['name']} (Max inputs: {device['max_input_channels']})")


if __name__ == "__main__":
    # Test the audio streamer
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--list-devices":
        list_audio_devices()
        sys.exit(0)
    
    # Simple test of audio streaming
    streamer = AudioStreamer()
    
    try:
        console.print("Testing audio streaming for 5 seconds...")
        streamer.start_recording()
        
        chunk_count = 0
        start_time = time.time()
        
        for wav_bytes, timestamp in streamer.get_audio_chunks():
            chunk_count += 1
            console.print(f"Got audio chunk {chunk_count}: {len(wav_bytes)} bytes at {timestamp:.2f}")
            
            # Stop after 5 seconds
            if time.time() - start_time > 5:
                break
        
        console.print(f"Captured {chunk_count} audio chunks")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    finally:
        streamer.cleanup() 