# Courtroom AI Assistant

A real-time transcription and legal response system designed for courtroom use. This application captures audio, transcribes speech using OpenAI Whisper, retrieves relevant legal documents, and generates concise responses within 4 seconds.

## 🎯 Core Features

- **Real-time Audio Capture**: Non-blocking microphone streaming with 3-second chunks
- **Fast Transcription**: OpenAI Whisper API integration with parallel processing
- **Document Retrieval**: FAISS-powered similarity search through legal documents
- **Intelligent Response Generation**: GPT-4o-mini generates ≤3 sentence responses
- **Speech Pattern Detection**: Automatically identifies questions, claims, objections, and citation requests
- **Turn-based Processing**: Smart silence detection for natural conversation flow

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Audio Stream   │───▶│   Transcription  │───▶│  Turn Detection │
│  (sounddevice)  │    │  (Whisper API)   │    │   (Silence +    │
└─────────────────┘    └──────────────────┘    │   Punctuation)  │
                                               └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│    Response     │◀───│   Document       │◀───│  Pattern        │
│   Generation    │    │   Retrieval      │    │  Detection      │
│  (GPT-4o-mini)  │    │  (FAISS Index)   │    │  (Rules-based)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.11 or higher
- OpenAI API key
- Microphone access
- Windows/macOS/Linux

### Setup

1. **Clone and Install Dependencies**
```bash
git clone <repository-url>
cd court_copilot
pip install -r requirements.txt
```

2. **Configure API Key**
```bash
# Option 1: Environment variable (recommended)
export OPENAI_API_KEY="your-api-key-here"

# Option 2: Edit config.toml
# Set openai_api_key = "your-api-key-here" in config.toml
```

3. **Prepare Documents**
```bash
# Create documents directory
mkdir docs

# Add your legal documents (PDF, Word, TXT files)
cp /path/to/legal/documents/* docs/
```

4. **Build Document Index**
```bash
# Index all documents in ./docs directory
python index_builder.py --docs ./docs --index ./index

# Custom chunking (optional)
python index_builder.py --docs ./docs --chunk-size 500 --chunk-overlap 75
```

## 🚀 Usage

### Basic Operation

```bash
# Start the assistant (live mode)
python main.py

# Test with a WAV file
python main.py --dry-run test.wav

# List available microphones
python main.py --list-devices

# Use custom configuration
python main.py --config custom.toml
```

### Example Session

```
🏛️ Courtroom AI Assistant initialized
📚 Index loaded: 156 chunks from 23 documents
🎤 Recording started
🚀 Courtroom AI Assistant is now active!

🎙️ Turn 1:
"What is the statute of limitations for breach of contract?"

🔍 Found 3 relevant chunks for query: 'What is the statute of limitations...'
💬 Response generated: 'Under UCC Section 2-725, the statute of limitations for breach of contract claims is generally 4 years from the date of breach. For contracts not governed by the UCC, most states follow a 6-year limitation period. The clock typically starts when the breach occurs, not when it's discovered.' (2.3s)

⏱️ Total processing time: 2.81s
```

## ⚙️ Configuration

Edit `config.toml` to customize behavior:

```toml
[api]
whisper_model = "whisper-1"          # Fastest Whisper model
chat_model = "gpt-4o-mini"           # Fast, cost-effective ChatGPT
embedding_model = "text-embedding-3-small"

[audio]
sample_rate = 16000                  # Optimal for Whisper
chunk_duration_seconds = 3.0         # Balance latency vs accuracy
silence_threshold_ms = 800           # Turn detection sensitivity

[response]
temperature = 0.2                    # Low randomness for consistency
max_tokens = 160                     # ~3 sentence limit
response_timeout_seconds = 4.0       # Hard timeout requirement

[retrieval]
max_retrieved_chunks = 5             # Context size vs relevance
similarity_threshold = 0.7           # Quality filter
```

## 🧪 Testing & Development

### Component Testing

```bash
# Test individual components
python audio_stream.py --list-devices  # Audio device testing
python transcribe.py                   # Transcription testing
python retriever.py                    # Document search testing
python responder.py                    # Response generation testing
```

### Dry-Run Testing

```bash
# Test with sample audio files
python main.py --dry-run samples/question.wav
python main.py --dry-run samples/objection.wav
```

### Unit Tests

```bash
# Run test suite (requires pytest)
pytest tests/
```

## 📊 Performance Metrics

The system is optimized for **<4 second response time**:

- **Audio Capture**: ~0ms (streaming)
- **Transcription**: 1-2s (parallel processing)
- **Document Retrieval**: 0.1-0.3s (FAISS indexing)
- **Response Generation**: 1-2s (GPT-4o-mini)
- **Total**: 2.5-3.5s typical

### Optimization Features

- **Parallel Processing**: Audio chunks processed simultaneously
- **Smart Chunking**: 300-token chunks with 50-token overlap
- **Efficient Indexing**: FAISS flat L2 index for speed
- **Minimal Latency**: Non-blocking audio streaming
- **Quick Fallback**: 2-second timeout responses available

## 🎯 Speech Pattern Recognition

The system automatically detects and responds to:

### Questions
- **Triggers**: "What", "When", "How", ending with "?"
- **Response**: Direct answers with citations

### Citation Requests  
- **Triggers**: "cite", "case law", "precedent", "authority"
- **Response**: Specific case names and statutes

### Claims/Arguments
- **Triggers**: "I submit", "clearly", "the evidence shows"
- **Response**: Counter-arguments and supporting precedent

### Objections
- **Triggers**: "objection", "hearsay", "relevance"
- **Response**: Procedural guidance and legal basis

## 🛡️ Security & Privacy

- **No Audio Storage**: Audio buffers are immediately zeroed after transcription
- **API Key Protection**: Keys loaded from environment variables only
- **Local Processing**: Document indexing happens locally
- **Minimal Data**: Only text transcripts sent to OpenAI APIs

## 🔧 Troubleshooting

### Common Issues

**No audio input detected**
```bash
# Check available devices
python main.py --list-devices

# Verify microphone permissions
# Windows: Privacy Settings → Microphone
# macOS: System Preferences → Security & Privacy → Microphone
```

**Transcription errors**
```bash
# Check API key
echo $OPENAI_API_KEY

# Test with simpler audio
python main.py --dry-run simple_test.wav
```

**No document results**
```bash
# Verify index exists
ls -la index/

# Rebuild index if needed
python index_builder.py --docs ./docs
```

**Slow response times**
- Check internet connection
- Reduce `max_retrieved_chunks` in config
- Use shorter audio chunks (reduce `chunk_duration_seconds`)

### Performance Tuning

For **faster responses** (target <3s):
```toml
chunk_duration_seconds = 2.5
max_retrieved_chunks = 3
max_tokens = 120
```

For **higher accuracy** (allow 4-5s):
```toml
chunk_duration_seconds = 4.0
max_retrieved_chunks = 7
max_tokens = 200
```

## 📁 File Structure

```
court_copilot/
├── main.py                  # Main application entry point
├── audio_stream.py          # Real-time audio capture
├── transcribe.py            # Whisper API integration
├── index_builder.py         # Document processing & indexing
├── retriever.py             # FAISS similarity search
├── responder.py             # GPT response generation
├── config.toml              # Configuration settings
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── docs/                   # Legal documents (PDFs, Word, TXT)
└── index/                  # Generated FAISS index files
    ├── faiss.index         # Vector index
    └── metadata.json       # Document metadata
```

## 🤝 Contributing

1. **Code Style**: Follow the existing patterns, use clear comments
2. **Testing**: Add tests for new features
3. **Performance**: Maintain <4s response time requirement
4. **Documentation**: Update README for new functionality

## 📄 License

This project is designed for legal professional use. Ensure compliance with local bar rules and ethical guidelines when using AI assistance in legal practice.

## 🆘 Support

For issues and questions:

1. Check this README's troubleshooting section
2. Review component test outputs
3. Verify configuration settings
4. Check OpenAI API status and quotas

---

**⚖️ Disclaimer**: This tool is designed to assist legal professionals but should not replace human judgment. Always verify AI-generated responses before use in legal proceedings. 