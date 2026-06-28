# Audio-First Podcast Pre-Editor

## Module 1: Media ingestion

The ingestion module scans a folder for `.mp4`, `.mov`, and `.wav` files. It
probes the original metadata, preserves available SMPTE/BWF timing metadata,
and converts every audio source to 16 kHz mono signed 16-bit PCM WAV.

Run it from the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.media.ingest input `
  --output-directory output\ingest
```

Use `--recursive` when media is stored in nested folders. The generated
`ingest_manifest.json` records:

- absolute and relative paths to each source asset;
- normalized WAV paths;
- exact rational frame rate and stream time base;
- duration, codec, dimensions, and audio properties;
- SMPTE timecode candidates, selected start timecode, and drop-frame status;
- Broadcast Wave `time_reference` samples when present.

If a source contains no SMPTE header, ingestion succeeds but records a null
timecode and prints a warning. Downstream timeline generation must not assume a
zero start timecode in this case.

Run the unit tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Module 2: Word-level transcription

Transcribe one normalized audio stem with an explicit speaker ID:

```powershell
.\.venv\Scripts\python.exe -m src.ai.speech.transcribe audio `
  output\ingest\audio\mic_host.wav.wav `
  --speaker-id Host `
  --output output\transcript_words.json
```

Transcribe all isolated `.wav` stems from the Module 1 manifest:

```powershell
.\.venv\Scripts\python.exe -m src.ai.speech.transcribe manifest `
  output\ingest\ingest_manifest.json `
  --speaker-map input\speaker_map.json `
  --output output\transcript_words.json
```

The optional speaker map is a JSON object. Keys can be manifest asset IDs,
relative source paths, source filenames, absolute source paths, or normalized
audio paths:

```json
{
  "asset_0002": "Host",
  "Guest Mic.wav": "Guest"
}
```

Without a speaker map, isolated WAV stems use their source filename as the
speaker ID. A single camera scratch track is labeled `unknown`; multiple camera
tracks require explicit `--asset-id` selections and a speaker map. Output is a
JSON array containing exactly `word`, `start_ms`, `end_ms`, and `speaker_id`.

## Module 3: Podcast Plan narrative filter

Configure an OpenAI-compatible chat-completions endpoint in `.env`:

```dotenv
LLM_API_URL=https://api.openai.com/v1/chat/completions
LLM_API_KEY=replace-with-your-key
LLM_MODEL=replace-with-a-structured-output-capable-model
```

Then filter the Module 2 transcript against a UTF-8 Podcast Plan:

```powershell
.\.venv\Scripts\python.exe -m src.analysis.rag_filter `
  input\podcast_plan.txt `
  output\transcript_words.json `
  --output output\keep_zones.json
```

The module converts individual words into timestamped speaker blocks, chunks
long transcripts before sending them to the LLM, validates that returned
timestamps stay inside their source chunks, and merges overlapping or adjacent
zones. The final contract is:

```json
{
  "keep_zones": [
    {"start_ms": 10000, "end_ms": 250000, "topic": "Introduction"}
  ]
}
```

### Local Hugging Face backend

The default local model is the official Qwen 2.5 3B Instruct GGUF `Q4_K_M`
file. Download it once (about 2.1 GB):

```powershell
.\.venv\Scripts\python.exe -m src.ai.llm.huggingface
```

Run Module 3 without an API key:

```powershell
.\.venv\Scripts\python.exe -m src.analysis.rag_filter `
  input\podcast_plan.txt `
  output\transcript_words.json `
  --backend huggingface `
  --output output\keep_zones.json
```

The checked-in dependency uses the small CPU llama.cpp wheel for reliable
Windows installation. If a CUDA llama-cpp-python wheel is installed later, set
`--hf-gpu-layers -1` to offload all layers that fit on the GPU. The local
backend defaults to 12,000 transcript characters per chunk and an 8,192-token
context to stay within the practical limits of a 3B model.
