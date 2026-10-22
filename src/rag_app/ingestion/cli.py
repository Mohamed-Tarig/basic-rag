import argparse
import json
import sys
from pathlib import Path

from rag_app.ingestion.pipeline import run_ingestion


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and activate a RAG document index."
    )

    parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Path to the PDF document",
    )

    parser.add_argument(
        "--collection-prefix",
        default="basic_rag",
        help="Qdrant collection prefix",
    )

    args = parser.parse_args()

    try:
        result = run_ingestion(
            pdf_path=args.file,
            collection_prefix=args.collection_prefix,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())