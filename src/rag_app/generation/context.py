from llama_index.core.schema import NodeWithScore


def build_context(nodes: list[NodeWithScore]) -> str:
    """Build a source-labelled context from reranked nodes."""

    if not nodes:
        return ""

    sections = []

    for index, item in enumerate(nodes, start=1):
        node = item.node

        source = node.metadata.get("file_name", "unknown")
        page = node.metadata.get("page_label", "unknown")
        content = node.get_content(metadata_mode="none").strip()

        if not content:
            continue

        sections.append(
            f"[Source {index}]\n"
            f"File: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{content}"
        )

    return "\n\n---\n\n".join(sections)