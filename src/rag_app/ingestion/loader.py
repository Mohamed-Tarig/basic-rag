from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader


def validate_document_path(path: Path) -> Path:
    """Validate that a PDF document exists."""
    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF documents are supported")

    return path

def load_documents(path: Path) -> list[Document]:
    """Load and parse a validated PDF document."""
    validated_path = validate_document_path(path)

    reader = SimpleDirectoryReader(
        input_files=[str(validated_path)],
        required_exts=[".pdf"],
    )

    return reader.load_data()