import re
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from PDF bytes.
    Indian court PDFs often have weird formatting —
    multiple spaces, broken lines, header/footer noise.
    We clean all of that here.
    """
    import io
    reader = PdfReader(io.BytesIO(file_bytes))
    
    pages = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        
        # clean up common Indian legal PDF issues
        # 1. remove page headers/footers (page numbers, court names repeated)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # 2. fix broken words across lines (hyphenation)
        text = re.sub(r'-\n([a-z])', r'\1', text)
        
        # 3. normalise whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        
        # 4. remove very short lines (usually noise)
        lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 20]
        text = '\n'.join(lines)
        
        pages.append(f"[Page {page_num + 1}]\n{text}")
    
    return '\n\n'.join(pages)


def chunk_legal_document(text: str) -> list[dict]:
    """
    Smart chunking for Indian legal documents.
    
    Instead of naive fixed-size chunks, we try to split on
    legal section boundaries first:
    - "HELD:" marks the court's decision
    - "JUDGMENT:" marks the start of reasoning  
    - Numbered paragraphs like "1.", "2.", "15."
    - "WHEREAS", "ORDER", "PETITION"
    
    This keeps legally meaningful units together.
    """
    
    # legal section markers specific to Indian courts
    legal_markers = [
        r'\nHELD\s*:',
        r'\nJUDGMENT\s*:',
        r'\nORDER\s*:',
        r'\nPETITION\s*:',
        r'\nFACTS\s*:',
        r'\nISSUE\s*:',
        r'\nREASONS\s*:',
        r'\nCONCLUSION\s*:',
        r'\n\d+\.\s+[A-Z]',   # numbered paragraphs
        r'\n\(\d+\)\s+',       # (1), (2), etc.
    ]
    
    # try to split on legal markers first
    pattern = '|'.join(legal_markers)
    sections = re.split(pattern, text)
    
    chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=['\n\n', '\n', '. ', ' ']
    )
    
    for i, section in enumerate(sections):
        if len(section.strip()) < 50:
            continue
            
        # if section is too long, split further
        if len(section) > 1000:
            sub_chunks = splitter.split_text(section)
            for j, sub in enumerate(sub_chunks):
                chunks.append({
                    "content": sub.strip(),
                    "section_idx": i,
                    "chunk_idx": j,
                    "metadata": extract_metadata(sub)
                })
        else:
            chunks.append({
                "content": section.strip(),
                "section_idx": i,
                "chunk_idx": 0,
                "metadata": extract_metadata(section)
            })
    
    return chunks


def extract_metadata(text: str) -> dict:
    """
    Extract useful metadata from a chunk.
    Used for filtering and display.
    """
    metadata = {}
    
    # detect if this chunk contains the court's decision
    if re.search(r'\b(HELD|DISMISSED|ALLOWED|UPHELD)\b', text, re.IGNORECASE):
        metadata['contains_judgment'] = True
    
    # detect case citations like "AIR 2019 SC 1234"
    citations = re.findall(r'AIR\s+\d{4}\s+[A-Z]+\s+\d+', text)
    if citations:
        metadata['citations'] = citations[:3]
    
    # detect section references like "Section 302 IPC"
    sections = re.findall(r'[Ss]ection\s+\d+[A-Z]?\s+[A-Z]+', text)
    if sections:
        metadata['law_sections'] = sections[:3]
    
    return metadata