import argparse
import json
import sys

from rag_app.generation.pipeline import generate_answer


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ask questions about indexed documents using RAG."
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Question to answer",
    )

    args = parser.parse_args()

    try:
        result = generate_answer(query=args.query)

        print(json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ))
        return 0

    except Exception as exc:  # noqa: BLE001
        print(f"RAG generation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())