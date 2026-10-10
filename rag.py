from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import requests
from docx import Document as WordDocument
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".pdf", ".docx"}


@dataclass
class Chunk:
    text: str
    source: str
    location: str


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def embeddings(self, texts: list[str], model: str) -> np.ndarray:
        response = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": model, "input": texts},
            timeout=self.timeout,
        )
        self._raise_for_status(response)
        return np.asarray(response.json()["embeddings"], dtype=np.float32)

    def chat(
        self, messages: list[dict[str, str]], model: str
    ) -> Iterator[str]:
        with requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": True,
                "think": False,
                "options": {"temperature": 0.2},
            },
            stream=True,
            timeout=self.timeout,
        ) as response:
            self._raise_for_status(response)
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as error:
            detail = response.text.strip()
            raise RuntimeError(
                f"Ollama request failed ({response.status_code}): {detail}"
            ) from error


def split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Split text near paragraph/sentence boundaries while retaining overlap."""
    cleaned = re.sub(r"[ \t]+", " ", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        target_end = min(start + chunk_size, len(cleaned))
        end = target_end
        if target_end < len(cleaned):
            search_from = start + chunk_size // 2
            candidates = [
                cleaned.rfind("\n\n", search_from, target_end),
                cleaned.rfind("。", search_from, target_end),
                cleaned.rfind(". ", search_from, target_end),
                cleaned.rfind("；", search_from, target_end),
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + (1 if cleaned[boundary] != "\n" else 0)

        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_document(path: Path, root: Path) -> list[Chunk]:
    relative_name = path.relative_to(root).as_posix()
    sections: list[tuple[str, str]] = []

    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        sections = [
            (page.extract_text() or "", f"page {number}")
            for number, page in enumerate(reader.pages, start=1)
        ]
    elif path.suffix.lower() == ".docx":
        document = WordDocument(path)
        sections = [
            ("\n".join(p.text for p in document.paragraphs), "document")
        ]
    else:
        sections = [(path.read_text(encoding="utf-8", errors="ignore"), "document")]

    result: list[Chunk] = []
    for section_text, location in sections:
        for index, text in enumerate(split_text(section_text), start=1):
            suffix = f"{location}, chunk {index}" if location != "document" else f"chunk {index}"
            result.append(Chunk(text=text, source=relative_name, location=suffix))
    return result


class LocalRAG:
    def __init__(
        self,
        documents_dir: Path,
        index_dir: Path,
        embedding_model: str = "bge-m3",
        chat_model: str = "qwen3:4b",
        ollama_url: str = "http://localhost:11434",
    ):
        self.documents_dir = documents_dir
        self.index_dir = index_dir
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.client = OllamaClient(ollama_url)
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None

    def document_files(self) -> list[Path]:
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        return sorted(
            path
            for path in self.documents_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for path in self.document_files():
            digest.update(path.relative_to(self.documents_dir).as_posix().encode())
            digest.update(path.read_bytes())
        digest.update(self.embedding_model.encode())
        return digest.hexdigest()

    def index_is_current(self) -> bool:
        metadata_path = self.index_dir / "metadata.json"
        vectors_path = self.index_dir / "vectors.npy"
        if not metadata_path.exists() or not vectors_path.exists():
            return False
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return metadata.get("fingerprint") == self.fingerprint()

    def build_index(self) -> int:
        files = self.document_files()
        if not files:
            raise ValueError(
                f"No supported documents found in {self.documents_dir}."
            )

        chunks = [
            chunk
            for path in files
            for chunk in load_document(path, self.documents_dir)
        ]
        if not chunks:
            raise ValueError("The documents did not contain any extractable text.")

        vectors = self.client.embeddings(
            [chunk.text for chunk in chunks], self.embedding_model
        )
        vectors = self._normalize(vectors)

        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.index_dir / "vectors.npy", vectors)
        metadata = {
            "fingerprint": self.fingerprint(),
            "embedding_model": self.embedding_model,
            "chunks": [asdict(chunk) for chunk in chunks],
        }
        (self.index_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.chunks = chunks
        self.vectors = vectors
        return len(chunks)

    def load_index(self) -> None:
        metadata_path = self.index_dir / "metadata.json"
        vectors_path = self.index_dir / "vectors.npy"
        if not metadata_path.exists() or not vectors_path.exists():
            raise FileNotFoundError("No index found. Build the index first.")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("embedding_model") != self.embedding_model:
            raise ValueError("The index was created with a different embedding model.")
        self.chunks = [Chunk(**item) for item in metadata["chunks"]]
        self.vectors = np.load(vectors_path)

    def search(self, question: str, top_k: int = 5) -> list[SearchResult]:
        if self.vectors is None:
            self.load_index()
        query = self._normalize(
            self.client.embeddings([question], self.embedding_model)
        )[0]
        scores = self.vectors @ query
        best = np.argsort(scores)[::-1][: min(top_k, len(scores))]
        return [
            SearchResult(chunk=self.chunks[int(index)], score=float(scores[index]))
            for index in best
        ]

    def answer(
        self, question: str, history: list[dict[str, str]] | None = None, top_k: int = 5
    ) -> tuple[Iterator[str], list[SearchResult]]:
        results = self.search(question, top_k)
        context = "\n\n".join(
            f"[{number}] Source: {result.chunk.source} ({result.chunk.location})\n"
            f"{result.chunk.text}"
            for number, result in enumerate(results, start=1)
        )
        system_prompt = (
            "You are a careful document assistant. Answer using only the supplied "
            "context. If the answer is not in the context, clearly say that the "
            "documents do not contain enough information. Cite factual claims with "
            "[1], [2], etc. Respond in the same language as the user's question.\n\n"
            f"CONTEXT:\n{context}"
        )
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend((history or [])[-6:])
        messages.append({"role": "user", "content": question})
        return self.client.chat(messages, self.chat_model), results

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)
