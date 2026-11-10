# Coverage Compass

Coverage Compass is a Week 2 RAG prototype that helps patients and nurses understand published benefits and prepare questions for the insurer. It supports specialist visits, outpatient rehabilitation, home health, and durable medical equipment for the 2026 Kaiser Permanente Covered California Silver 70 HMO.

## Run locally

Use Python 3.11 or later. In PowerShell, open this project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000. The default local mode needs no credentials. For optional model-assisted retrieval and answer selection, copy `.env.example` to `.env` and set your own OpenAI API key. Use fictional questions with this prototype. Never commit `.env`. The optional mode sends the question and retrieved policy summaries to the model provider.

```powershell
.\.venv\Scripts\python.exe evaluate.py
```

## How it works

1. The policy corpus contains 11 human-curated, paraphrased sections and benefit table rows with stable IDs, topic labels, source pages, and retrieval terms.
2. LangChain indexes the documents in an in-memory vector store. Local mode uses deterministic hashed word and character features. Optional model-assisted mode uses OpenAI embeddings.
3. BM25 keyword search and vector similarity are fused by reciprocal rank. The graph keeps up to six passages and reserves a slot for each recognized service.
4. LangGraph classifies the question and conditionally runs retrieval. It routes to cited policy evidence, a clarification, insurer confirmation, or an out-of-scope message.
5. Local generation formats retrieved facts into patient explanations or nurse checklists. In model-assisted mode, the model chooses source IDs under the prompt in `engine.py`. The answer renderer uses only curated policy text and links associated with those IDs.

The UI is a local prototype with a fixed policy snapshot and no connection to member records. It explains published rules and directs individual questions to Member Services.

## Reproduce the live API demo

With Ollama running, pull the demo model once. Then launch the app with the environment setting below. The model API runs on this computer and uses no API key.

```powershell
ollama pull llama3.2:3b
$env:CC_OLLAMA_MODEL = "llama3.2:3b"
.\.venv\Scripts\python.exe app.py
```

Run the nurse discharge example and expand **Inspect retrieval and routing** to see the service routing, hybrid order, and evidence in the answer. The Ollama badge appears when the model returns a usable selection. The answer preserves the other retrieved passages so relevant policy conditions remain visible. Clear `CC_OLLAMA_MODEL` when reproducing the bundled local evaluation.

## Project materials

- `PROJECT.md` is the write-up to paste into Google Docs.
- `evaluation-results.json` contains the 15 cases, expected evidence, actual routes, and measured retrieval and latency.
- `data/policy.json` contains the policy snapshot used for retrieval.

Upload this source folder to your own GitHub repository for submission. Keep the demo video and Google Doc as separate submission links.
