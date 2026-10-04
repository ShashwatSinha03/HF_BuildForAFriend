"""Document extraction and retrieval utilities for StudyLens."""

import re
from pypdf import PdfReader
from typing import Optional


CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MAX_CHUNKS_FOR_CONTEXT = 3
MAX_CHUNKS_FOR_REVISION = 6
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "during",
    "before", "after", "above", "below", "between", "among", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "must", "can",
    "this", "that", "these", "those", "i", "you", "he", "she", "it", "we",
    "they", "what", "which", "who", "whom", "whose", "where", "when", "why",
    "how", "all", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "don", "should", "now", "if", "then", "else", "when", "while"
}


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract readable text from a PDF file."""
    import io
    reader = PdfReader(io.BytesIO(file_bytes))
    text_parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)
    return "\n".join(text_parts)


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Read text directly from a TXT file."""
    return file_bytes.decode("utf-8")


def extract_text(file_bytes: bytes, file_type: str) -> str:
    """Extract text from file based on its type."""
    if file_type == "pdf":
        return extract_text_from_pdf(file_bytes)
    elif file_type == "txt":
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def get_file_type(filename: str) -> Optional[str]:
    """Determine file type from filename extension."""
    ext = filename.lower().split(".")[-1]
    if ext in ("pdf", "txt"):
        return ext
    return None


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, remove punctuation."""
    return re.sub(r"[^\w\s]", " ", text.lower())


def _tokenize(text: str) -> list[str]:
    """Extract meaningful tokens from text (preserves duplicates for frequency)."""
    words = _normalize_text(text).split()
    return [w for w in words if len(w) > 2 and w not in STOP_WORDS]


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by character count."""
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []
    
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def retrieve_relevant_chunks(
    document_text: str,
    question: str,
    max_chunks: int = MAX_CHUNKS_FOR_CONTEXT,
) -> list[str]:
    """
    Retrieve most relevant chunks for a question using token frequency overlap.
    Returns up to max_chunks most relevant chunks.
    """
    if not document_text.strip() or not question.strip():
        return []
    
    chunks = split_into_chunks(document_text)
    if not chunks:
        return []
    
    question_tokens = _tokenize(question)
    if not question_tokens:
        return chunks[:max_chunks]
    
    question_token_set = set(question_tokens)
    
    scored_chunks = []
    for chunk in chunks:
        chunk_tokens = _tokenize(chunk)
        if not chunk_tokens:
            continue
        
        chunk_token_set = set(chunk_tokens)
        
        # Count unique question tokens found in chunk
        matched_tokens = question_token_set & chunk_token_set
        unique_match_count = len(matched_tokens)
        
        if unique_match_count == 0:
            continue
        
        # Count total frequency of matched tokens in chunk
        total_freq = sum(1 for t in chunk_tokens if t in matched_tokens)
        
        # Score: unique matches primary, frequency secondary
        # This prioritizes chunks with more distinct question terms
        # and breaks ties with frequency
        score = unique_match_count * 100 + total_freq
        
        scored_chunks.append((score, chunk))
    
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored_chunks[:max_chunks]]


def get_representative_chunks(
    document_text: str,
    max_chunks: int = MAX_CHUNKS_FOR_REVISION,
) -> list[str]:
    """
    Select representative chunks from across the document for revision.
    Uses a simple strategy: distribute selection across the document
    and prefer chunks with higher information density (more unique tokens).
    """
    if not document_text.strip():
        return []
    
    chunks = split_into_chunks(document_text)
    if not chunks:
        return []
    
    # If document is small, use all chunks
    if len(chunks) <= max_chunks:
        return chunks
    
    # Score chunks by information density (unique token count)
    scored_chunks = []
    for i, chunk in enumerate(chunks):
        tokens = _tokenize(chunk)
        unique_tokens = len(set(tokens))
        # Boost first and last chunks slightly (often contain intro/summary)
        position_boost = 0
        if i == 0 or i == len(chunks) - 1:
            position_boost = 5
        score = unique_tokens + position_boost
        scored_chunks.append((score, i, chunk))
    
    # Sort by score descending
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    # Take top chunks, but ensure some distribution across the document
    selected = []
    selected_indices = set()
    
    for score, idx, chunk in scored_chunks:
        if len(selected) >= max_chunks:
            break
        # Avoid selecting too many adjacent chunks
        adjacent = any(abs(idx - si) <= 1 for si in selected_indices)
        if not adjacent or len(selected) < max_chunks // 2:
            selected.append((idx, chunk))
            selected_indices.add(idx)
    
    # Sort selected by original document order
    selected.sort(key=lambda x: x[0])
    return [chunk for _, chunk in selected]