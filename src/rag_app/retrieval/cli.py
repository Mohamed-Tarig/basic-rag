import argparse
import json
import sys

from rag_app.retrieval.pipeline import retrieve_and_rerank


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retrieve and rerank relevant document chunks."
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Question to search for",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of chunks after reranking",
    )

    args = parser.parse_args()

    try:
        results = retrieve_and_rerank(
            query=args.query,
            retrieval_top_k=args.top_k,
            rerank_top_n=args.top_n,
        )

        output = [
            {
                "rank": rank,
                "score": item.score,
                "source": item.node.metadata.get("file_name"),
                "page": item.node.metadata.get("page_label"),
                "text": item.node.get_content(metadata_mode="none"),
            }
            for rank, item in enumerate(results, 1)
        ]

        print(json.dumps(output, indent=2, ensure_ascii=False))
        return 0

    except Exception as exc:  # noqa: BLE001
        print(f"Retrieval failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())