"""
Bank Officer Policy Management & Document Ingestion UI Module.

Research Title:
An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation
Using Machine Learning Predictions

Module Purpose:
- Provides Bank Officers with a graphical interface to upload, validate, index, and manage institutional policy files (PDF, DOCX, TXT).
- Maintains strict provenance, SHA-256 hash deduplication, and lifecycle status governance (active, superseded, archived).
- Provides an interactive policy retrieval verification panel.
"""

from datetime import date
from pathlib import Path
import tempfile
from typing import Optional

import streamlit as st

from rag.config import default_rag_config
from rag.document_processor import DocumentProcessor
from rag.embedding_service import get_embedding_service
from rag.ingest import DocumentIngestionManager
from rag.retriever import PolicyRetriever
from rag.vector_store import ChromaVectorStore


@st.cache_resource
def get_rag_components():
    """Initialize and cache shared RAG components."""
    config = default_rag_config
    processor = DocumentProcessor(config)
    embedding_service = get_embedding_service(config)
    vector_store = ChromaVectorStore(config, embedding_service)
    ingest_mgr = DocumentIngestionManager(config, processor, vector_store)
    retriever = PolicyRetriever(config, vector_store)
    return config, processor, embedding_service, vector_store, ingest_mgr, retriever


def render_policy_management_page():
    """Render the full Policy Management dashboard for Bank Officers."""
    config, processor, embedding_service, vector_store, ingest_mgr, retriever = get_rag_components()

    st.markdown('<div class="main-header">📚 Bank Policy Management & Knowledge Base</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        '<strong>Institutional Document-Driven Policy Ingestion & Governance</strong>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="advisory-banner">'
        '<strong>📋 INSTITUTIONAL POLICY UPLOAD NOTICE:</strong> '
        'Only authorized institutional policy documents should be uploaded. '
        'Policy content is treated as evidence for decision support and is not hardcoded into application logic. '
        'The system remains policy-neutral until authorized documents are uploaded.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Tabs for Organization
    tab_upload, tab_list, tab_search = st.tabs([
        "📥 Upload New Policy Document",
        "📚 Indexed Policy Documents",
        "🔎 Test Policy Retrieval",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: UPLOAD NEW POLICY DOCUMENT
    # -------------------------------------------------------------------------
    with tab_upload:
        st.markdown("### 📥 Upload Institutional Policy Document")
        st.caption("Upload verified bank credit policies, underwriting circulars, or regulatory guidelines (PDF, DOCX, TXT).")

        with st.form("policy_upload_form", clear_on_submit=True):
            uploaded_file = st.file_uploader(
                "Select Policy File (Max 15MB) *",
                type=["pdf", "docx", "txt"],
                help="Supported formats: PDF (.pdf), Microsoft Word (.docx), Plain Text (.txt).",
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                policy_version = st.text_input("Policy Version *", value="1.0", help="e.g. 1.0, 2026.Q1, Circular-42/2025")
            with c2:
                effective_date_val = st.date_input("Effective Date *", value=date.today())
            with c3:
                status_val = st.selectbox("Policy Status *", options=["active", "superseded", "archived"], index=0)

            policy_desc = st.text_area(
                "Optional Policy Description / Scope",
                placeholder="e.g. Retail banking secured vehicle loan guidelines and collateral requirements.",
                help="Brief internal note for credit officers.",
            )

            upload_submit = st.form_submit_button("📥 Upload & Index Policy Document", type="primary", use_container_width=True)

        if upload_submit:
            if uploaded_file is None:
                st.error("❌ Please select a valid policy document file before submitting.")
            else:
                # Save temporarily for processing
                with tempfile.TemporaryDirectory() as tmp_dir:
                    temp_path = Path(tmp_dir) / uploaded_file.name
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    with st.spinner(f"⏳ Processing and indexing '{uploaded_file.name}' into ChromaDB..."):
                        result = ingest_mgr.ingest_file(
                            file_path=temp_path,
                            document_name=uploaded_file.name.rsplit(".", 1)[0].replace("_", " ").title(),
                            policy_version=policy_version.strip() or "1.0",
                            effective_date=str(effective_date_val),
                            status=status_val,
                            description=policy_desc.strip() if policy_desc else None,
                            copy_to_storage=True,
                        )

                    if result["success"]:
                        st.success(
                            f"✅ **Policy Successfully Indexed!**\n\n"
                            f"- **Document ID:** `{result['document_id']}`\n"
                            f"- **Document Name:** `{result['document_name']}`\n"
                            f"- **Total Chunks Created:** `{result['total_chunks']}`\n"
                            f"- **Chunks Indexed:** `{result['chunks_indexed']}`\n"
                            f"- **Status:** `{status_val.upper()}`"
                        )
                    elif result.get("is_duplicate"):
                        st.warning(
                            f"⚠️ **Duplicate Document Detected:**\n\n"
                            f"{result['error']} (Matching Document ID: `{result.get('document_id')}`)."
                        )
                    else:
                        st.error(f"❌ **Ingestion Failed:** {result.get('error', 'Unknown error.')}")

    # -------------------------------------------------------------------------
    # TAB 2: INDEXED POLICY DOCUMENTS LIST
    # -------------------------------------------------------------------------
    with tab_list:
        st.markdown("### 📚 Indexed Policy Documents")
        st.caption("Active and historical policy documents stored in the ChromaDB vector database.")

        docs = ingest_mgr.list_documents()

        if not docs:
            st.warning("⚠️ **Policy Knowledge Base Empty**")
            st.markdown(
                "> *No institutional policy documents have been supplied. Policy-grounded verification cannot be completed.*"
            )
            st.caption("Bank officers may upload real documents using the 'Upload New Policy Document' tab.")
        else:
            st.info(f"📊 **Total Registered Policy Documents:** `{len(docs)}` | **Vector Chunks in ChromaDB:** `{vector_store.get_collection_stats()['total_chunks_indexed']}`")

            for doc in docs:
                doc_id = doc["document_id"]
                doc_name = doc.get("document_name", "Untitled Document")
                doc_type = doc.get("document_type", "UNKNOWN")
                version = doc.get("policy_version", "1.0")
                eff_date = doc.get("effective_date", "Not specified")
                status = doc.get("status", "active")
                chunks_count = doc.get("total_chunks", 0)
                upload_ts = doc.get("upload_timestamp", "N/A")[:10]

                # Status color badge
                if status == "active":
                    status_badge = "🟢 ACTIVE"
                elif status == "superseded":
                    status_badge = "🟡 SUPERSEDED"
                else:
                    status_badge = "⚪ ARCHIVED"

                with st.expander(f"📄 {doc_name} (v{version}) — {status_badge}"):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"**Document ID:** `{doc_id}`")
                        st.markdown(f"**Format:** `{doc_type}`")
                        st.markdown(f"**Chunks:** `{chunks_count}`")
                    with c2:
                        st.markdown(f"**Policy Version:** `{version}`")
                        st.markdown(f"**Effective Date:** `{eff_date}`")
                        st.markdown(f"**Uploaded:** `{upload_ts}`")
                    with c3:
                        st.markdown(f"**Current Status:** `{status.upper()}`")
                        st.markdown(f"**File Hash (SHA-256):** `{doc.get('file_hash', '')[:12]}...`")

                    if doc.get("description"):
                        st.markdown(f"**Description:** *{doc['description']}*")

                    st.markdown("---")
                    st.markdown("**Officer Lifecycle Actions:**")
                    b_col1, b_col2, b_col3 = st.columns(3)
                    with b_col1:
                        if status != "active":
                            if st.button(f"🟢 Set Active", key=f"act_{doc_id}"):
                                ingest_mgr.set_document_status(doc_id, "active")
                                st.success(f"Status updated to ACTIVE for {doc_name}")
                                st.rerun()
                    with b_col2:
                        if status != "superseded":
                            if st.button(f"🟡 Mark Superseded", key=f"sup_{doc_id}"):
                                ingest_mgr.set_document_status(doc_id, "superseded")
                                st.info(f"Status updated to SUPERSEDED for {doc_name}")
                                st.rerun()
                    with b_col3:
                        if status != "archived":
                            if st.button(f"⚪ Mark Archived", key=f"arc_{doc_id}"):
                                ingest_mgr.set_document_status(doc_id, "archived")
                                st.warning(f"Status updated to ARCHIVED for {doc_name}")
                                st.rerun()

    # -------------------------------------------------------------------------
    # TAB 3: TEST POLICY RETRIEVAL (VERIFICATION PANEL)
    # -------------------------------------------------------------------------
    with tab_search:
        st.markdown("### 🔎 Test Policy Retrieval (Verification Panel)")
        st.caption("Verify semantic search and inspect retrieved policy clauses with exact similarity scores and page provenance.")

        search_col1, search_col2 = st.columns([3, 1])
        with search_col1:
            search_query = st.text_input(
                "Enter Policy Query:",
                placeholder="e.g. What are the collateral requirements for business loans above 5 million?",
            )
        with search_col2:
            top_k = st.slider("Top Chunks (k)", min_value=1, max_value=5, value=3)

        filter_active = st.checkbox("Filter Active Policies Only (Exclude Superseded/Archived)", value=True)

        if st.button("🔍 Search Policy Knowledge Base", type="primary", use_container_width=True):
            if not search_query.strip():
                st.warning("⚠️ Please enter a search query.")
            else:
                with st.spinner("Searching ChromaDB vector store..."):
                    results = retriever.retrieve(
                        query=search_query,
                        top_k=top_k,
                        filter_active_only=filter_active,
                    )

                if not results:
                    st.info("ℹ️ No matching policy clauses found in the active knowledge base.")
                else:
                    st.success(f"✅ Found {len(results)} relevant policy clauses.")
                    st.markdown("#### **Retrieved Institutional Policy Evidence:**")

                    for idx, res in enumerate(results, 1):
                        doc_name = res.get("document_name", "Unknown Document")
                        page = res.get("page_number", -1)
                        page_str = f"Page {page}" if page != -1 else "N/A"
                        score = res.get("relevance_score", 0.0)
                        version = res.get("policy_version", "1.0")
                        status = res.get("status", "unknown")
                        chunk_id = res.get("chunk_id", "unknown")
                        text_snippet = res.get("text", "").strip()

                        with st.expander(f"📄 [Result {idx}] {doc_name} (v{version}, {page_str}) — Score: {score:.4f} [{status.upper()}]"):
                            st.markdown(f"**Chunk ID:** `{chunk_id}` | **Similarity Score:** `{score:.4f}`")
                            st.markdown(f"**Policy Excerpt:**\n> {text_snippet}")
