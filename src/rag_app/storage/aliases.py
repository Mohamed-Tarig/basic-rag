from qdrant_client import QdrantClient
from qdrant_client.http import models


def activate_collection(
    client: QdrantClient,
    collection_name: str,
    alias_name: str = "basic_rag_active",
) -> None:
    """Atomically activate a collection through a Qdrant alias."""

    if not client.collection_exists(collection_name):
        raise ValueError(f"Collection does not exist: {collection_name}")

    aliases = client.get_aliases().aliases

    current = next(
        (a for a in aliases if a.alias_name == alias_name),
        None,
    )

    if current and current.collection_name == collection_name:
        return

    operations = []

    if current:
        operations.append(
            models.DeleteAliasOperation(
                delete_alias=models.DeleteAlias(
                    alias_name=alias_name,
                )
            )
        )

    operations.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(
                collection_name=collection_name,
                alias_name=alias_name,
            )
        )
    )

    client.update_collection_aliases(
        change_aliases_operations=operations
    )