"""
Document Ingestion & Registry Management Orchestration.

Research Title:
An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation
Using Machine Learning Predictions

Module Purpose:
- Orchestrates the end-to-end ingestion lifecycle for policy documents (PDF, DOCX, TXT).
- Enforces SHA-256 hash deduplication to prevent redundant indexing.
- Maintains the authoritative document registry in rag/metadata/documents_registry.json.
- Manages document status lifecycle (active, superseded, archived) in both registry and ChromaDB.
"""

import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional

from rag.config import RAGConfig, default_rag_config
from rag.document_processor import DocumentProcessor
from rag.vector_store import ChromaVectorStore


class DocumentIngestionManager:
    """Manages document uploads, processing, vector store ingestion, and metadata tracking."""

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        processor: Optional[DocumentProcessor] = None,
        vector_store: Optional[ChromaVectorStore] = None,
    ):
        self.config = config or default_rag_config
        self.processor = processor or DocumentProcessor(self.config)
        self.vector_store = vector_store or ChromaVectorStore(self.config)
        self._ensure_registry_exists()

    def _ensure_registry_exists(self):
        """Create empty registry file if not present."""
        if not self.config.REGISTRY_FILE.exists():
            with open(self.config.REGISTRY_FILE, "w", encoding="utf-8") as f:
                json.dump({"documents": {}, "total_documents": 0}, f, indent=2)

    def _load_registry(self) -> Dict[str, Any]:
        """Load document metadata registry from JSON."""
        self._ensure_registry_exists()
        try:
            with open(self.config.REGISTRY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"documents": {}, "total_documents": 0}

    def _save_registry(self, registry: Dict[str, Any]):
        """Save document metadata registry to JSON."""
        with open(self.config.REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)

    def get_existing_hashes(self) -> Dict[str, str]:
        """Map of file_hash -> document_id for existing registered documents."""
        registry = self._load_registry()
        hashes: Dict[str, str] = {}
        for doc_id, meta in registry.get("documents", {}).items():
            f_hash = meta.get("file_hash")
            if f_hash:
                hashes[f_hash] = doc_id
        return hashes

    def ingest_file(
        self,
        file_path: Path,
        document_name: Optional[str] = None,
        policy_version: str = "1.0",
        effective_date: Optional[str] = None,
        status: str = "active",
        description: Optional[str] = None,
        copy_to_storage: bool = True,
    ) -> Dict[str, Any]:
        """Ingest a policy file: validate, deduplicate, store, extract, chunk, embed, and index."""
        is_valid, msg = self.processor.validate_file(file_path)
        if not is_valid:
            return {"success": False, "error": msg, "is_duplicate": False}

        # 1. SHA-256 Duplicate Check
        file_hash = self.processor.compute_file_hash(file_path)
        existing_hashes = self.get_existing_hashes()
        if file_hash in existing_hashes:
            existing_doc_id = existing_hashes[file_hash]
            return {
                "success": False,
                "is_duplicate": True,
                "error": "Document already exists in the policy repository.",
                "document_id": existing_doc_id,
            }

        # 2. Optionally copy original document to rag/documents/ storage
        target_path = file_path
        if copy_to_storage and file_path.parent != self.config.DOCUMENTS_DIR:
            dest_file = self.config.DOCUMENTS_DIR / file_path.name
            # If a file with same name exists, add unique suffix
            if dest_file.exists() and self.processor.compute_file_hash(dest_file) != file_hash:
                dest_file = self.config.DOCUMENTS_DIR / f"{file_path.stem}_{file_hash[:8]}{file_path.suffix}"
            shutil.copy2(str(file_path), str(dest_file))
            target_path = dest_file

        # 3. Process text into chunks
        try:
            processed_data = self.processor.process_document(
                file_path=target_path,
                document_name=document_name or file_path.stem.replace("_", " ").title(),
                policy_version=policy_version,
                effective_date=effective_date,
                status=status,
            )
        except Exception as e:
            return {"success": False, "error": f"Text processing failed: {str(e)}", "is_duplicate": False}

        summary = processed_data["summary"]
        chunks = processed_data["chunks"]

        if description:
            summary["description"] = description.strip()

        # 4. Insert chunks into ChromaDB
        try:
            chunks_indexed = self.vector_store.add_chunks(chunks)
        except Exception as e:
            return {"success": False, "error": f"Vector indexing failed: {str(e)}", "is_duplicate": False}

        # 5. Update registry
        registry = self._load_registry()
        doc_id = summary["document_id"]
        registry["documents"][doc_id] = {
            **summary,
            "chunks_indexed": chunks_indexed,
        }
        registry["total_documents"] = len(registry["documents"])
        self._save_registry(registry)

        return {
            "success": True,
            "is_duplicate": False,
            "document_id": doc_id,
            "document_name": summary["document_name"],
            "total_chunks": len(chunks),
            "chunks_indexed": chunks_indexed,
            "summary": summary,
        }

    def list_documents(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all indexed documents, optionally filtered by status."""
        registry = self._load_registry()
        docs = list(registry.get("documents", {}).values())
        if status_filter:
            docs = [d for d in docs if d.get("status") == status_filter]
        return docs

    def get_document_metadata(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a specific document ID."""
        registry = self._load_registry()
        return registry.get("documents", {}).get(document_id)

    def set_document_status(self, document_id: str, new_status: str) -> bool:
        """Update status of a document (e.g. 'active', 'superseded', 'archived') in both registry and ChromaDB."""
        registry = self._load_registry()
        if document_id not in registry.get("documents", {}):
            return False

        # 1. Update status in registry
        registry["documents"][document_id]["status"] = new_status
        self._save_registry(registry)

        # 2. Synchronize status across chunks in ChromaDB
        self.vector_store.update_document_status(document_id, new_status)
        return True

    def delete_document(self, document_id: str, delete_source_file: bool = False) -> bool:
        """Remove document chunks from ChromaDB and delete from registry."""
        registry = self._load_registry()
        if document_id not in registry.get("documents", {}):
            return False

        doc_meta = registry["documents"][document_id]

        # 1. Delete from vector store
        self.vector_store.delete_by_document_id(document_id)

        # 2. Optionally delete source file from rag/documents/
        if delete_source_file:
            source_file = self.config.DOCUMENTS_DIR / doc_meta.get("source_file", "")
            if source_file.exists():
                try:
                    source_file.unlink()
                except Exception:
                    pass

        # 3. Update registry
        del registry["documents"][document_id]
        registry["total_documents"] = len(registry["documents"])
        self._save_registry(registry)

        return True
