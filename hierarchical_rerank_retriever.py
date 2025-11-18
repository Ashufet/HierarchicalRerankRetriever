from typing import Any, List, Optional

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.embeddings.utils import resolve_embed_model
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import (
    Document,
    IndexNode,
    TextNode,
    BaseNode,
    NodeWithScore,
    QueryBundle,
)
from llama_index.core.node_parser.text.utils import split_by_sentence_tokenizer
from llama_index.core.base.base_retriever import BaseRetriever
from collections import Counter
from typing import List, Any


def top_k_by_frequency_then_first_index(values: List[Any], top_k: int = 5) -> List[Any]:
    """
    Return the top-k values from 'values' based on:
      1. Frequency (descending)
      2. Earliest occurrence in 'values' (ascending index) if frequencies tie.

    Args:
        values: A list that may contain duplicates.
        top_k (int, optional): Number of items to return. Defaults to 5.

    Returns:
        A list of up to 'top_k' items satisfying the above criteria.
    """
    # 1) Count frequency of each item
    freq_map = Counter(values)
    
    # 2) Determine the first occurrence (index) of each item
    #    (We only record the first time we see each item.)
    first_occurrence_map = {}
    for idx, item in enumerate(values):
        if item not in first_occurrence_map:
            first_occurrence_map[item] = idx

    # 3) Sort items primarily by descending frequency,
    #    then by ascending first-occurrence index (earliest).
    sorted_items = sorted(
        freq_map.items(),
        key=lambda x: (-x[1], first_occurrence_map[x[0]])
    )

    # 4) Return just the top_k item values (not frequencies)
    return [item for item, _ in sorted_items[:top_k]]


class HierarchicalRerankRetriever(BaseRetriever):
    """
    A BaseRetriever subclass that:
      1) Splits documents into base chunks (~base_chunk_size).
      2) Splits each base chunk into intermediate sub-chunks (~intermediate_chunk_size).
      3) Further splits sub-chunks into sentences.
      4) Builds two VectorStoreIndexes:
         - `sent_vector_index` (sentence-level)
         - `other_vector_index` (sub-chunk + base-chunk)
      5) Retrieval process:
         - Retrieve top matches from `sent_vector_index`.
         - Map matches back to sub-chunk nodes in `other_vector_index`.
         - Optionally rerank sub-chunks using a `reranker`.
         - Return the final nodes (wrapped in `NodeWithScore`).
    """

    def __init__(
        self,
        docs: Optional[List[Document]] = None,
        reranker: Optional[BaseNodePostprocessor] = None,
        embed_model: str = "local:BAAI/bge-small-en",
        base_chunk_size: int = 2048,
        base_chunk_overlap: int = 200,
        intermediate_chunk_size: int = 512,
        intermediate_chunk_overlap: int = 0,
        similarity_top_k: int = 5,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the hierarchical rerank retriever.

        Args:
            docs (List[Document], optional): The documents to index.
            reranker (BaseNodePostprocessor, optional): Custom reranker implementing `postprocess_nodes()`.
            embed_model (str, optional): Embedding model reference. Defaults to "local:BAAI/bge-small-en".
            base_chunk_size (int, optional): Chunk size for the initial base chunks (default 2048).
            base_chunk_overlap (int, optional): Overlap size for base chunks (default 0).
            intermediate_chunk_size (int, optional): Chunk size for intermediate chunks (default 512).
            intermediate_chunk_overlap (int, optional): Overlap size for intermediate chunks (default 0).
            similarity_top_k (int, optional): Number of top sentences retrieved from sentence-level index (default 5).
            verbose (bool, optional): Whether to print verbose logs. Defaults to False.
            **kwargs (Any): Additional arguments passed to the BaseRetriever or other components.
        """
        # Initialize BaseRetriever internals (callback_manager, object_map, etc.)
        super().__init__(verbose=verbose, **kwargs)

        self.similarity_top_k = similarity_top_k

        # 1) Build base chunks
        base_parser = SentenceSplitter(
            chunk_size=base_chunk_size, 
            chunk_overlap=base_chunk_overlap
        )
        self.base_nodes = base_parser.get_nodes_from_documents(docs or [])

        # Assign unique IDs to each base node
        for idx, node in enumerate(self.base_nodes):
            node.id_ = f"node-{idx}"

        # 2) Set embedding model globally for vector indexes
        self.embed_model = resolve_embed_model(embed_model)
        Settings.embed_model = self.embed_model

        # 3) Store the provided reranker
        self.reranker = reranker

        # 4) Build intermediate chunks + sentence-level nodes
        sub_parser = SentenceSplitter(
            chunk_size=intermediate_chunk_size, 
            chunk_overlap=intermediate_chunk_overlap
        )
        sent_tokenizer = split_by_sentence_tokenizer()

        all_sub_nodes: List[IndexNode] = []
        all_sent_nodes: List[IndexNode] = []

        # For each base chunk, build sub-chunks and their sentence-level chunks
        for base_node in self.base_nodes:
            sub_nodes = sub_parser.get_nodes_from_documents([base_node])
            # Convert each sub-chunk to an IndexNode referencing the base node
            for sub_node in sub_nodes:
                sub_index_node = IndexNode.from_text_node(sub_node, base_node.node_id)
                all_sub_nodes.append(sub_index_node)

                # Sentence-level splitting
                sentences = sent_tokenizer(sub_node.text)
                for s_text in sentences:
                    sent_index_node = IndexNode.from_text_node(
                        TextNode(text=s_text), 
                        sub_node.node_id
                    )
                    all_sent_nodes.append(sent_index_node)

        # 5) Build three vector indexes:
        #    - Sentence-level index
        self.sent_vector_index = VectorStoreIndex(all_sent_nodes, show_progress=True)
        #    - Sub node index
        self.sub_vector_index = VectorStoreIndex(all_sub_nodes, show_progress=True)
        #    - base chunk index
        self.base_vector_index = VectorStoreIndex(self.base_nodes, show_progress=True)

        # 6) Create a retriever for the sentence-level index and intermediate-level index
        self.sent_vector_retriever = self.sent_vector_index.as_retriever(
            similarity_top_k=self.similarity_top_k
        )

        self.sub_vector_retriever = self.sub_vector_index.as_retriever(
            similarity_top_k=self.similarity_top_k
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Implements the `_retrieve` method from BaseRetriever.

        Steps:
          1. Retrieve top sentences from `sent_vector_retriever`.
          2. Map them back to sub-chunk nodes in `other_vector_index`.
          3. Rerank if `self.reranker` is provided.
          4. Return final nodes, wrapped in NodeWithScore.

        Args:
            query_bundle (QueryBundle): Contains the query string and any metadata.

        Returns:
            List[NodeWithScore]: The list of final nodes, each wrapped in NodeWithScore.
        """

        # 1) Retrieve top sentence-level matches (with scores)
        sentence_nodes_with_scores = self.sent_vector_retriever.retrieve(query_bundle.query_str)

        # 2) Extract their sub-chunk node IDs
        sub_nodes_with_scores = self.sub_vector_retriever.retrieve(query_bundle.query_str)
        
        sub_node_ids = [nwscore.node.node_id for nwscore in sub_nodes_with_scores]
        sub_node_ids.extend([nwscore.node.index_id for nwscore in sentence_nodes_with_scores])

        # 3) Fetch unique sub-chunk nodes from our "other_vector_index"
        sub_nodes = self.sub_vector_index.docstore.get_nodes(top_k_by_frequency_then_first_index(sub_node_ids, self.similarity_top_k))
        sub_nodes_with_score = [NodeWithScore(node=n) for n in sub_nodes]

        # 4) If a reranker is provided, rerank the sub-chunk nodes
        if self.reranker:
            sub_nodes_with_score = self.reranker.postprocess_nodes(
                sub_nodes_with_score, query_bundle
            )

        # 5) Convert final sub_nodes_with_score to final node IDs
        final_node_ids = [nwscore.node.index_id for nwscore in sub_nodes_with_score]

        # Fetch unique final nodes from the docstore
        final_nodes = self.base_vector_index.docstore.get_nodes(list(dict.fromkeys(final_node_ids)))

        # sub_nodes_with_score is a list of NodeWithScore objects.
        # We want to build a dictionary {index_id -> score}, preserving the first occurrence.
        final_node_id_to_score = {}
        for node_with_score in sub_nodes_with_score:
            node_id = node_with_score.node.index_id
            # Only set score if index_id not already present,
            # preserving the "first" score in case of duplicates.
            if node_id not in final_node_id_to_score:
                final_node_id_to_score[node_id] = node_with_score.score
        
        # Return them as NodeWithScore with sub-chunk node scores
        final_nodes_with_score = []
        for fn in final_nodes:
            score = final_node_id_to_score.get(fn.node_id, None)
            final_nodes_with_score.append(NodeWithScore(node=fn, score=score))

        return final_nodes_with_score
    

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Asynchronous version of the `_retrieve` method from BaseRetriever.

        Steps:
            1. Retrieve top sentence-level matches (async).
            2. Map them back to sub-chunk nodes in `other_vector_index` (possibly async).
            3. Rerank if `self.reranker` is provided and has an async method.
            4. Return final nodes, wrapped in NodeWithScore.
        """
        # 1) Retrieve top sentence-level matches (with scores)
        sentence_nodes_with_scores = await self.sent_vector_retriever.aretrieve(query_bundle.query_str)

        # 2) Extract their sub-chunk node IDs
        sub_nodes_with_scores = await self.sub_vector_retriever.aretrieve(query_bundle.query_str)
        
        sub_node_ids = [nwscore.node.node_id for nwscore in sub_nodes_with_scores]
        sub_node_ids.extend([nwscore.node.index_id for nwscore in sentence_nodes_with_scores])

        # 3) Fetch unique sub-chunk nodes from our "other_vector_index"
        sub_nodes = await self.sub_vector_index.docstore.aget_nodes(top_k_by_frequency_then_first_index(sub_node_ids, self.similarity_top_k))
        sub_nodes_with_score = [NodeWithScore(node=n) for n in sub_nodes]

        
        # 4) If a reranker is provided, rerank the sub-chunk nodes
        if self.reranker:
            sub_nodes_with_score = self.reranker.postprocess_nodes(
                sub_nodes_with_score, query_bundle
            )

        # 5) Convert final sub_nodes_with_score to final node IDs
        final_node_ids = [nwscore.node.index_id for nwscore in sub_nodes_with_score]

        # Fetch unique final nodes from the docstore
        final_nodes = await self.base_vector_index.docstore.aget_nodes(list(dict.fromkeys(final_node_ids)))

        # sub_nodes_with_score is a list of NodeWithScore objects.
        # We want to build a dictionary {index_id -> score}, preserving the first occurrence.
        final_node_id_to_score = {}
        for node_with_score in sub_nodes_with_score:
            node_id = node_with_score.node.index_id
            # Only set score if index_id not already present,
            # preserving the "first" score in case of duplicates.
            if node_id not in final_node_id_to_score:
                final_node_id_to_score[node_id] = node_with_score.score
        
        # Return them as NodeWithScore with sub-chunk node scores
        final_nodes_with_score = []
        for fn in final_nodes:
            score = final_node_id_to_score.get(fn.node_id, None)
            final_nodes_with_score.append(NodeWithScore(node=fn, score=score))

        return final_nodes_with_score
