from unittest.mock import Mock

import pytest
from qdrant_client.http import models

from rag_app.storage.aliases import activate_collection


def test_rejects_missing_collection():
    client = Mock()
    client.collection_exists.return_value = False

    with pytest.raises(ValueError):
        activate_collection(client, "missing_collection")

    client.update_collection_aliases.assert_not_called()


def test_creates_alias_when_missing():
    client = Mock()
    client.collection_exists.return_value = True
    client.get_aliases.return_value.aliases = []

    activate_collection(client, "rag_v1")

    operations = client.update_collection_aliases.call_args.kwargs[
        "change_aliases_operations"
    ]

    assert len(operations) == 1
    assert isinstance(operations[0], models.CreateAliasOperation)
    assert operations[0].create_alias.collection_name == "rag_v1"


def test_switches_existing_alias():
    client = Mock()
    client.collection_exists.return_value = True
    client.get_aliases.return_value.aliases = [
        Mock(alias_name="basic_rag_active", collection_name="rag_v1")
    ]

    activate_collection(client, "rag_v2")

    operations = client.update_collection_aliases.call_args.kwargs[
        "change_aliases_operations"
    ]

    assert len(operations) == 2
    assert isinstance(operations[0], models.DeleteAliasOperation)
    assert isinstance(operations[1], models.CreateAliasOperation)
    assert operations[1].create_alias.collection_name == "rag_v2"


def test_activation_is_idempotent():
    client = Mock()
    client.collection_exists.return_value = True
    client.get_aliases.return_value.aliases = [
        Mock(alias_name="basic_rag_active", collection_name="rag_v2")
    ]

    activate_collection(client, "rag_v2")

    client.update_collection_aliases.assert_not_called()