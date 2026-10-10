# Local RAG Demo

A small, fully local document question-answering app for live demonstrations. It
uses Chinese-developed open models through Ollama:

- **Qwen3 4B** for answers — small enough for a modern MacBook Pro while still
  supporting Chinese and English well.
- **BGE-M3** for multilingual document retrieval.

Documents and conversations remain on the laptop. Internet access is only
needed initially to install software and download the models.

## 1. Install Ollama

Download and install Ollama from [ollama.com/download](https://ollama.com/download),
then open Terminal and download the models:

```bash
ollama pull qwen3:4b
ollama pull bge-m3
```

Keep Ollama running. Both models together require roughly 4–5 GB of disk space.

## 2. Set up the app

From this project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## 3. Add documents

Put files in the `documents` folder. Supported formats:

- PDF (`.pdf`)
- Microsoft Word (`.docx`)
- Text and Markdown (`.txt`, `.md`)
- CSV and JSON (`.csv`, `.json`)

Scanned/image-only PDFs need OCR first; this demo extracts embedded text only.

## 4. Run

```bash
streamlit run app.py
```

Open the URL shown in Terminal, click **Build / refresh index**, and ask a
question. The app automatically detects when source documents have changed.

## Demo tips

1. Start with 3–10 short, text-based documents that the audience recognizes.
2. Ask one question with a precise answer in the documents.
3. Expand **Sources used** to show retrieval and citations.
4. Ask something absent from the documents to demonstrate that the assistant
   should say it lacks enough information.
5. Ask the same question in Chinese and English to demonstrate multilingual RAG.

## Example questions

These questions work with the sample files included in `documents`:

- How many days of annual leave do full-time employees receive?
- What approvals do I need for an international trip costing RMB 10,000?
- Compare KnowledgeBox and PredictFlow, including their prices.
- What must be completed before a customer project can go live?
- If a production issue occurs, how quickly must the response group be created?
- Which offices does the company have, and how many employees are there?
- 员工每年可以申请多少培训预算？
- KnowledgeBox 的价格是多少？它会把数据发送到公共云吗？
- 标准客户项目需要多长时间？每个阶段分别需要多久？
- 如果我要去上海出差，酒店每晚的报销上限是多少？

For a useful “not found” demonstration, ask:

- What is the CEO's phone number?
- 公司食堂今天的菜单是什么？

The assistant should explain that the documents do not contain this information
instead of inventing an answer.

For an 8 GB Mac, change the chat model in the sidebar to `qwen3:1.7b` after
running `ollama pull qwen3:1.7b`. For 16 GB or more, `qwen3:4b` is the recommended
demo default.

## How it works

1. The app extracts text and splits it into overlapping chunks.
2. BGE-M3 converts each chunk into an embedding stored in `.rag_index`.
3. A question is embedded and matched against the most similar chunks.
4. Qwen receives only those chunks and writes an answer with citations.
