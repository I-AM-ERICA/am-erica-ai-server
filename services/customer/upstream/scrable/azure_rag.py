"""Optional Azure AI Search adapter matching the slide's deployed architecture."""

import os
import re
from dataclasses import asdict

from .io import digest
from .types import Document


class AzureRetriever:
    def __init__(self, chunks, embedder, index_prefix="scrable"):
        from azure.core.credentials import AzureKeyCredential
        from azure.core.exceptions import ResourceNotFoundError
        from azure.search.documents import SearchClient
        from azure.search.documents.indexes import SearchIndexClient
        from azure.search.documents.indexes.models import (
            HnswAlgorithmConfiguration,
            SearchableField,
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            SimpleField,
            VectorSearch,
            VectorSearchProfile,
        )

        if not re.fullmatch(r"[a-z][a-z0-9-]{0,40}", index_prefix):
            raise ValueError("azure_index_prefix must be lowercase letters, numbers or hyphens")
        endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
        credential = AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"])
        self.embedder = embedder
        self.fingerprint = digest({"chunks": [asdict(c) for c in chunks], "embedding": embedder.identity})
        # Content-addressed index: never mix documents from different splits/runs.
        self.index_name = f"{index_prefix}-{self.fingerprint[:16]}"
        admin = SearchIndexClient(endpoint, credential)
        self.client = SearchClient(endpoint, self.index_name, credential)
        try:
            admin.get_index(self.index_name)
        except ResourceNotFoundError:
            dim = len(embedder.encode([chunks[0].text])[0])
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(name="text", type=SearchFieldDataType.String),
                SimpleField(name="app_id", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="source", type=SearchFieldDataType.String),
                SearchField(
                    name="vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=dim,
                    vector_search_profile_name="scrable-vector",
                ),
            ]
            admin.create_index(
                SearchIndex(
                    name=self.index_name,
                    fields=fields,
                    vector_search=VectorSearch(
                        algorithms=[HnswAlgorithmConfiguration(name="scrable-hnsw")],
                        profiles=[
                            VectorSearchProfile(
                                name="scrable-vector", algorithm_configuration_name="scrable-hnsw"
                            )
                        ],
                    ),
                )
            )
        # Upsert even when index exists: recovers from interrupted partial uploads.
        for start in range(0, len(chunks), 64):
            batch = chunks[start : start + 64]
            vectors = embedder.encode([c.text for c in batch])
            results = self.client.merge_or_upload_documents(
                [{**asdict(c), "vector": v.tolist()} for c, v in zip(batch, vectors)]
            )
            errors = [r.error_message for r in results if not r.succeeded]
            if errors:
                raise RuntimeError(f"Azure indexing failed: {errors}")

    def retrieve(self, query, app_id, top_k=4):
        from azure.search.documents.models import VectorizedQuery

        vector = self.embedder.encode([query])[0].tolist()
        results = self.client.search(
            search_text=query,
            vector_queries=[
                VectorizedQuery(vector=vector, k_nearest_neighbors=max(top_k, 20), fields="vector")
            ],
            filter="app_id eq '" + app_id.replace("'", "''") + "'",
            select=["id", "text", "app_id", "source"],
            top=top_k,
        )
        return [Document(**{k: row[k] for k in ("id", "text", "app_id", "source")}) for row in results]
