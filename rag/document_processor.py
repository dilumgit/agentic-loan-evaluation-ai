"""
Document Processing, Text Extraction, and Chunking Pipeline.

Research Title:
An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation
Using Machine Learning Predictions

Module Purpose:
- Validates untrusted external document uploads (PDF, DOCX, TXT).
- Extracts clean textual content with accurate provenance (page numbers where available).
- Performs deterministic, configurable text chunking with full audit metadata.
- Prepares structured chunk objects for vector store indexing.
"""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import docx
import pypdf

from rag.config import RAGConfig, default_rag_config


class DocumentProcessor:
    """Handles secure extraction, cleaning, chunking, and metadata tagging for bank policy files."""

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or default_rag_config

    def validate_file(self, file_path: Path) -> Tuple[bool, str]:
        """Verify file existence, extension, and size within security bounds."""
        if not file_path.exists():
            return False, f"File does not exist: {file_path}"

        ext = file_path.suffix.lower()
        if ext not in self.config.ALLOWED_EXTENSIONS:
            return False, f"Unsupported file type '{ext}'. Allowed: {sorted(list(self.config.ALLOWED_EXTENSIONS))}"

        file_size = file_path.stat().st_size
        if file_size == 0:
            return False, "File is empty (0 bytes)."
        if file_size > self.config.MAX_FILE_SIZE_BYTES:
            return False, f"File size ({file_size / (1024*1024):.2f} MB) exceeds limit of {self.config.MAX_FILE_SIZE_BYTES / (1024*1024):.1f} MB."

        return True, "Validation successful."

    def compute_file_hash(self, file_path: Path) -> str:
        """Generate SHA-256 hash of document binary for deduplication and provenance."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def clean_text(self, text: str) -> str:
        """Sanitize and normalize extracted text without executing embedded macros or scripts."""
        if not text:
            return ""
        # Remove null bytes and non-printable control characters (except standard whitespace)
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        # Normalize whitespace (replace multiple spaces/tabs with single space, preserve paragraph newlines)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)
        return cleaned.strip()

    def extract_text(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract text blocks with page/section provenance based on file format."""
        is_valid, msg = self.validate_file(file_path)
        if not is_valid:
            raise ValueError(f"Document validation failed: {msg}")

        ext = file_path.suffix.lower()
        extracted_pages: List[Dict[str, Any]] = []

        if ext == ".pdf":
            reader = pypdf.PdfReader(str(file_path))
            for page_idx, page in enumerate(reader.pages, 1):
                try:
                    raw_text = page.extract_text() or ""
                    cleaned = self.clean_text(raw_text)
                    if cleaned:
                        extracted_pages.append({
                            "page_number": page_idx,
                            "text": cleaned,
                        })
                except Exception as e:
                    # Log page extraction issue and continue with remaining pages
                    continue

        elif ext == ".docx":
            doc = docx.Document(str(file_path))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            full_text = "\n\n".join(paragraphs)
            cleaned = self.clean_text(full_text)
            if cleaned:
                extracted_pages.append({
                    "page_number": None,  # DOCX format does not have deterministic physical pages
                    "text": cleaned,
                })

        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()
            cleaned = self.clean_text(raw_text)
            if cleaned:
                extracted_pages.append({
                    "page_number": None,
                    "text": cleaned,
                })

        return extracted_pages

    def split_text_into_chunks(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[str]:
        """Deterministic, recursive text chunker splitting on paragraph and sentence boundaries."""
        size = chunk_size or self.config.CHUNK_SIZE
        overlap = chunk_overlap or self.config.CHUNK_OVERLAP

        if not text or len(text) <= size:
            return [text] if text and len(text) >= self.config.MIN_CHUNK_LENGTH else []

        chunks: List[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + size
            if end >= text_len:
                chunk = text[start:].strip()
                if len(chunk) >= self.config.MIN_CHUNK_LENGTH:
                    chunks.append(chunk)
                break

            # Find natural break point near end (prefer double newline, newline, period, space)
            candidate_segment = text[start:end]
            split_idx = -1

            for sep in ["\n\n", "\n", ". ", "; ", ", ", " "]:
                idx = candidate_segment.rfind(sep)
                if idx != -1 and idx >= int(size * 0.4):  # Don't make chunks too small
                    split_idx = idx + len(sep)
                    break

            if split_idx == -1:
                split_idx = size  # hard break if no natural boundary

            chunk = text[start : start + split_idx].strip()
            if len(chunk) >= self.config.MIN_CHUNK_LENGTH:
                chunks.append(chunk)

            # Advance with overlap
            start = start + max(1, split_idx - overlap)

        return chunks

    def process_document(
        self,
        file_path: Path,
        document_id: Optional[str] = None,
        document_name: Optional[str] = None,
        policy_version: str = "1.0",
        effective_date: Optional[str] = None,
        status: str = "active",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Full pipeline: extract, clean, chunk, and attach complete metadata to each chunk."""
        is_valid, msg = self.validate_file(file_path)
        if not is_valid:
            raise ValueError(msg)

        file_hash = self.compute_file_hash(file_path)
        doc_id = document_id or f"doc_{file_hash[:12]}"
        doc_name = document_name or file_path.name
        doc_type = file_path.suffix.lower().replace(".", "").upper()
        upload_ts = datetime.now(timezone.utc).isoformat()

        extracted_blocks = self.extract_text(file_path)
        if not extracted_blocks:
            raise ValueError(f"No textual content could be extracted from {file_path.name}.")

        all_chunks: List[Dict[str, Any]] = []
        global_chunk_idx = 1

        for block in extracted_blocks:
            page_num = block.get("page_number")
            raw_text = block["text"]

            sub_chunks = self.split_text_into_chunks(
                raw_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            for chunk_text in sub_chunks:
                chunk_id = f"{doc_id}_chunk_{global_chunk_idx:04d}"
                chunk_meta = {
                    "document_id": doc_id,
                    "document_name": doc_name,
                    "document_type": doc_type,
                    "source_file": str(file_path.name),
                    "file_hash": file_hash,
                    "page_number": page_num if page_num is not None else -1,
                    "chunk_id": chunk_id,
                    "chunk_index": global_chunk_idx,
                    "upload_timestamp": upload_ts,
                    "policy_version": policy_version,
                    "effective_date": effective_date or "Not specified",
                    "status": status,
                }
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "metadata": chunk_meta,
                })
                global_chunk_idx += 1

        document_summary = {
            "document_id": doc_id,
            "document_name": doc_name,
            "document_type": doc_type,
            "source_file": str(file_path.name),
            "file_hash": file_hash,
            "file_size_bytes": file_path.stat().st_size,
            "total_extracted_blocks": len(extracted_blocks),
            "total_chunks": len(all_chunks),
            "upload_timestamp": upload_ts,
            "policy_version": policy_version,
            "effective_date": effective_date or "Not specified",
            "status": status,
        }

        return {
            "summary": document_summary,
            "chunks": all_chunks,
        }
