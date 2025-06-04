# 🏛️ Courtroom AI Assistant - Complete Setup Guide

This guide walks you through setting up and using the Courtroom AI Assistant **as if you're preparing for and sitting in court**.

## 📋 **Pre-Court Preparation (Night Before/Morning Of)**

### Step 1: Secure Installation & API Key Setup

```bash
# 1. Clone and install (do this once)
git clone <your-repo-url>
cd court_copilot
pip install -r requirements.txt

# 2. Secure API key setup (NEVER push this to GitHub)
python setup_api_key.py
```

When you run the setup script, it will:
- ✅ Create a `.env` file (ignored by git)
- ✅ Securely store your OpenAI API key
- ✅ Test the connection

### Step 2: Prepare Your Legal Documents

```bash
# Create docs directory if it doesn't exist
mkdir -p docs

# Copy your case materials (DO NOT push sensitive docs to GitHub!)
cp /path/to/case/documents/* docs/
cp /path/to/relevant/statutes/* docs/
cp /path/to/precedent/cases/* docs/
```

**Supported document types:**
- 📄 **PDFs**: Case law, statutes, contracts, evidence
- 📄 **Word docs**: Briefs, motions, depositions
- 📄 **Text files**: Notes, summaries, research

### Step 3: Build Document Index

```bash
# Index all your legal documents (this takes a few minutes)
python index_builder.py --docs ./docs --index ./index

# You should see output like:
# ✅ Processed: contract_dispute_cases.pdf (2,456 tokens)
# ✅ Processed: state_statutes.docx (1,789 tokens)
# 📚 Index loaded: 156 chunks from 23 documents
```

### Step 4: Test Your Setup

```bash
# Test microphone and basic functionality
python main.py --list-devices

# Test with sample speech (create a test WAV file)
python main.py --dry-run test_question.wav
```

---

## 🎯 **Courtroom Day - Live Usage**

### Setting Up in Court

**Hardware Setup:**
1. **Laptop Position**: Place laptop between you and co-counsel
2. **Microphone**: Use built-in mic or external USB mic near speaking area
3. **Power**: Ensure laptop is plugged in (this is CPU-intensive)
4. **Internet**: Stable connection required for OpenAI API calls

**Before Proceedings Start:**

```bash
# 1. Navigate to your case directory
cd /path/to/case_name_court_ai

# 2. Start the assistant
python main.py

# You'll see:
# 🏛️ Courtroom AI Assistant initialized
# 📚 Index loaded: 156 chunks from 23 documents  
# 🎤 Recording started
# 🚀 Courtroom AI Assistant is now active!
# Listening for speech... Press Ctrl+C to stop.
```

### Real-Time Usage Examples

#### **Scenario 1: Opposition Makes a Legal Claim**

**What You Hear:**
> *"Your Honor, the plaintiff clearly cannot meet the burden of proof under the preponderance standard established in Smith v. Jones."*

**System Response:**
```
🎙️ Turn 1:
"Your Honor, the plaintiff clearly cannot meet the burden of proof under the preponderance standard established in Smith v. Jones."

🔍 Found 3 relevant chunks for query: 'burden of proof preponderance Smith v. Jones'
⚖️ Legal Response (claim, 2.3s)
The burden of proof in civil cases requires only a preponderance of evidence, meaning more likely than not (51%). Smith v. Jones actually established that circumstantial evidence can satisfy this standard. Consider citing Martinez v. State (2019) which clarifies that reasonable inferences from available evidence are sufficient.
```

**What You Do:**
- Quickly read the response to co-counsel
- Prepare your rebuttal using the Martinez citation
- Ask for clarification on the Smith v. Jones reference

---

#### **Scenario 2: Judge Asks a Complex Question**

**What You Hear:**
> *"Counsel, what is the statute of limitations for breach of fiduciary duty claims in this jurisdiction?"*

**System Response:**
```
🎙️ Turn 2:
"Counsel, what is the statute of limitations for breach of fiduciary duty claims in this jurisdiction?"

🔍 Found 2 relevant chunks for query: 'statute of limitations breach fiduciary duty'
❓ Legal Response (question, 1.8s)
Under State Code Section 12.3-401, fiduciary duty breach claims have a 4-year statute of limitations from discovery of the breach. However, if fraud is involved, the period extends to 6 years under the discovery rule established in Anderson v. Trust Corp.
```

**What You Do:**
- Stand and respond: *"Your Honor, the statute of limitations is 4 years from discovery under State Code Section 12.3-401..."*
- Mention the 6-year extension for fraud if relevant to your case

---

#### **Scenario 3: Need Quick Citation**

**What You Hear:**
> *"I need the case that established the reasonable person standard in negligence."*

**System Response:**
```
🎙️ Turn 3:
"I need the case that established the reasonable person standard in negligence."

🔍 Found 1 relevant chunk for query: 'reasonable person standard negligence case'
📚 Legal Response (citation_request, 1.2s)
The reasonable person standard was established in Vaughan v. Menlove (1837). The court held that the standard is objective, not subjective, asking what a reasonable person would do in similar circumstances.
```

**What You Do:**
- Whisper to co-counsel: "Vaughan v. Menlove, 1837"
- Prepare to cite the objective standard rule

---

#### **Scenario 4: Opposition Objects**

**What You Hear:**
> *"Objection, Your Honor! Hearsay!"*

**System Response:**
```
🎙️ Turn 4:
"Objection, Your Honor! Hearsay!"

🔍 Found 2 relevant chunks for query: 'objection hearsay'
🚫 Legal Response (objection, 1.5s)
Consider responding with hearsay exceptions under FRE 803/804. If the statement falls under present sense impression, excited utterance, or business records exceptions, argue admissibility. Present the specific exception that applies.
```

**What You Do:**
- Quickly identify which hearsay exception applies
- Respond: *"Your Honor, this falls under the business records exception..."*

---

## ⚙️ **Optimizing for Court Performance**

### Performance Tuning for Court

**For Maximum Speed (< 3 seconds):**
```toml
# Edit config.toml
[audio]
chunk_duration_seconds = 2.5

[retrieval]
max_retrieved_chunks = 3

[response]
max_tokens = 120
```

**For Maximum Accuracy (3-4 seconds):**
```toml
[audio]
chunk_duration_seconds = 3.5

[retrieval]
max_retrieved_chunks = 5

[response]
max_tokens = 160
```

### Courtroom Etiquette

**DO:**
- ✅ Keep laptop screen angled away from judge/jury
- ✅ Read responses quietly to co-counsel
- ✅ Use responses as research starting points, not verbatim
- ✅ Verify citations before using them
- ✅ Practice with the system beforehand

**DON'T:**
- ❌ Read AI responses directly to the court
- ❌ Rely solely on AI without verification
- ❌ Let the system distract from active listening
- ❌ Use in jurisdictions where AI assistance is prohibited

---

## 🚨 **Troubleshooting During Court**

### Quick Fixes

**No response generated:**
```bash
# System may be overloaded, restart quickly
Ctrl+C
python main.py
```

**Slow responses:**
- Check internet connection
- Reduce `max_retrieved_chunks` to 3
- Close other applications

**Microphone issues:**
```bash
# Quick device check
python main.py --list-devices
```

**API errors:**
- Check if you have OpenAI credits remaining
- Verify internet connection
- Use quick response mode (shorter, faster responses)

### Emergency Backup

If the system fails during court:
1. **Continue manually** - you're still a lawyer first
2. **Quick restart** - Ctrl+C and restart during break
3. **Fallback to notes** - your prepared materials are still there

---

## 📊 **Post-Court Analysis**

After court, review the session statistics:

```
📊 Session Statistics:
  Duration: 127.3s
  Turns processed: 8
  Responses generated: 6
  Avg processing time per turn: 2.1s

📝 Transcription Statistics:
  Total requests: 12
  Success rate: 91.7%
  Real-time factor: 1.2x

💬 Response Generation Statistics:
  Total requests: 6
  Success rate: 100.0%
  Avg response time: 2.1s
```

This helps you understand:
- How well the system performed
- Which queries worked best
- Areas for improvement in your document index

---

## 🔒 **Security Reminders**

**Before Court:**
- ✅ Ensure `.env` file is never pushed to git
- ✅ Verify sensitive documents aren't in public repositories
- ✅ Test with dummy data first

**After Court:**
- ✅ Audio is never stored (automatically cleared)
- ✅ Only text transcripts sent to OpenAI
- ✅ Consider clearing transcription logs if sensitive

**Compliance:**
- ✅ Check local bar rules about AI assistance
- ✅ Some jurisdictions require disclosure
- ✅ Always verify AI-generated responses independently

---

**⚖️ Remember: This tool assists your legal expertise; it doesn't replace your professional judgment. You remain responsible for all arguments and citations used in court.** 