# src/rag/cultural.py
# ---------------------------------------------------------------------------
# Optional heavy dependencies.
# If *any* of them fail to import (e.g. NumPy ABI mismatch), fall back to a
# lightweight stub so the rest of the pipeline can still run.
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional


_HAS_EMBED_DEPS = True  # sentence-transformers + numpy + pandas
_HAS_FAISS = True

try:
    import pandas as pd  # type: ignore
    import numpy as np  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
except Exception as _e:  # pragma: no cover
    print(f"⚠️  Core deps missing for cultural retrieval ({_e}).")
    _HAS_EMBED_DEPS = False

# FAISS is optional – we can fall back to pure-NumPy search.
if _HAS_EMBED_DEPS:
    try:
        import faiss  # type: ignore
    except Exception:  # pragma: no cover – FAISS often unavailable on Apple Silicon
        _HAS_FAISS = False

# Try instantiating the model early so we surface NumPy / torch issues quickly.
if _HAS_EMBED_DEPS:
    try:
        _EMB = SentenceTransformer("BAAI/bge-small-en-v1.5")
    except Exception as _e:  # pragma: no cover
        print(f"⚠️  sentence-transformers unusable ({_e}); cultural retrieval disabled.")
        _HAS_EMBED_DEPS = False

# Overall flag: we can proceed if embedding deps are present (FAISS optional)
_HAS_DEPS = _HAS_EMBED_DEPS

# ---------------------------------------------------------------------------
# Full or partial implementation depending on FAISS availability
# ---------------------------------------------------------------------------

if _HAS_DEPS:
    EMB = _EMB  # promote to module-level constant

    class CulturalRetriever:  # noqa: D101 – docstring kept short
        """Cultural retriever using FAISS-HNSW index."""

        def __init__(self, index_path: Optional[Path] = None):
            """
            Initialize cultural retriever.
            
            Args:
                index_path: Path to pre-built FAISS index (optional)
            """
            self.index = None
            self.snippets: List[str] = []
            self.metadata: List[Dict] = []
            self.embeddings = None
            self.is_loaded = False
            
            if index_path and index_path.exists():
                self.load_index(index_path)
        
        def build_index(self, snippets_file: Path, output_dir: Path) -> None:
            """
            Build FAISS-HNSW index from cultural snippets.
            
            Args:
                snippets_file: TSV file with cultural snippets
                output_dir: Directory to save index files
            """
            print("Loading cultural snippets...")
            
            # Load snippets from TSV
            df = pd.read_csv(snippets_file, sep='\t')
            
            # Extract text and metadata
            self.snippets = df['text'].tolist()
            self.metadata = df.to_dict('records')
            
            print(f"Loaded {len(self.snippets)} cultural snippets")
            
            # Compute embeddings
            print("Computing embeddings...")
            self.embeddings = EMB.encode(
                self.snippets,
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
            
            if _HAS_FAISS:
                # ---------------- FAISS path ----------------
                print("Building FAISS Flat-IP index (cosine similarity)...")
                dimension = self.embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                self.index.add(self.embeddings.astype("float32"))
                print(f"Index built with {self.index.ntotal} vectors (FAISS)")
            else:
                # ---------------- NumPy fallback ----------------
                print("FAISS not available – using in-memory NumPy index (dot-product)...")
                # For NumPy fallback we just keep self.embeddings; 'index' remains None.
                self.index = None
            
            # Save index and metadata
            self.save_index(output_dir)
        
        def save_index(self, output_dir: Path) -> None:
            """Save FAISS index and metadata."""
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, str(output_dir / "cultural_index.faiss"))
            
            # Save metadata
            with open(output_dir / "cultural_metadata.json", 'w') as f:
                json.dump({
                    'snippets': self.snippets,
                    'metadata': self.metadata,
                    'embedding_shape': self.embeddings.shape
                }, f, indent=2)
            
            print(f"Index saved to {output_dir}")
        
        def load_index(self, index_path: Path) -> None:
            """Load pre-built FAISS index."""
            index_dir = index_path.parent if index_path.suffix == '.faiss' else index_path
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_dir / "cultural_index.faiss"))
            
            # Load metadata
            with open(index_dir / "cultural_metadata.json", 'r') as f:
                data = json.load(f)
                self.snippets = data['snippets']
                self.metadata = data['metadata']
            
            self.is_loaded = True
            print(f"Loaded index with {self.index.ntotal} vectors")
        
        def query(self, query_text: str, top_k: int = 5, 
                  min_similarity: float = 0.27) -> List[Dict]:
            """
            Query cultural snippets.
            
            Args:
                query_text: Query text
                top_k: Number of results to return
                min_similarity: Minimum similarity threshold
                
            Returns:
                List of cultural snippets with similarity scores
            """
            if not self.is_loaded and self.embeddings is None:
                raise ValueError("Index not loaded. Call load_index() first.")

            # Compute query embedding
            query_embedding = EMB.encode([query_text], normalize_embeddings=True)

            if _HAS_FAISS and self.index is not None:
                # -------- FAISS search --------
                similarities, indices = self.index.search(
                    query_embedding.astype("float32"),
                    top_k,
                )
            else:
                # -------- NumPy fallback search --------
                similarities = cosine_similarity(query_embedding, self.embeddings)[0]
                # Get top_k indices
                indices = np.argsort(similarities)[::-1][:top_k]
                similarities = similarities[indices]
                indices = indices.reshape(1, -1)
            
            # Filter by similarity threshold and format results
            results = []
            for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
                if sim >= min_similarity:
                    result = {
                        'snippet': self.snippets[idx],
                        'similarity': float(sim),
                        'rank': i + 1,
                        'metadata': self.metadata[idx] if idx < len(self.metadata) else {}
                    }
                    results.append(result)
            
            return results
        
        def query_ctu(self, ctu_text: str, ctu_lang_counts: Dict[str, int], 
                      top_k: int = 5) -> List[Dict]:
            """
            Query cultural snippets for a CTU.
            
            Args:
                ctu_text: CTU text
                ctu_lang_counts: Language distribution in CTU
                top_k: Number of results to return
                
            Returns:
                List of cultural snippets relevant to the CTU
            """
            # Create query from CTU text and dominant language
            dominant_lang = max(ctu_lang_counts.items(), key=lambda x: x[1])[0]
            
            # Add language context to query
            query = f"{ctu_text} [language: {dominant_lang}]"
            
            return self.query(query, top_k)
        
        def batch_query(self, queries: List[str], top_k: int = 5) -> List[List[Dict]]:
            """
            Batch query multiple texts.
            
            Args:
                queries: List of query texts
                top_k: Number of results per query
                
            Returns:
                List of result lists
            """
            if not self.is_loaded and self.embeddings is None:
                raise ValueError("Index not loaded. Call load_index() first.")

            # Compute embeddings for all queries
            query_embeddings = EMB.encode(
                queries,
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True,
            )

            if _HAS_FAISS and self.index is not None:
                similarities, indices = self.index.search(
                    query_embeddings.astype("float32"),
                    top_k,
                )
            else:
                # NumPy fallback – compute full cosine sim matrix
                sim_matrix = cosine_similarity(query_embeddings, self.embeddings)
                indices = np.argsort(sim_matrix, axis=1)[:, ::-1][:, :top_k]
                # Gather similarities into same-shape array
                similarities = np.take_along_axis(sim_matrix, indices, axis=1)
            
            # Format results
            all_results = []
            for query_idx, (sims, idxs) in enumerate(zip(similarities, indices)):
                query_results = []
                for i, (sim, idx) in enumerate(zip(sims, idxs)):
                    if sim >= 0.27:  # min_similarity threshold
                        result = {
                            'snippet': self.snippets[idx],
                            'similarity': float(sim),
                            'rank': i + 1,
                            'metadata': self.metadata[idx] if idx < len(self.metadata) else {}
                        }
                        query_results.append(result)
                all_results.append(query_results)
            
            return all_results

if not _HAS_DEPS:
    class CulturalRetriever:  # noqa: D101 – minimal stub fallback
        """Stub implementation used when heavy optional dependencies are missing.

        This allows the rest of the application to import `CulturalRetriever` even
        if numpy / sentence-transformers / faiss etc. are not available. All
        retrieval-related methods degrade gracefully by returning empty results
        instead of raising at import-time.
        """

        def __init__(self, *_, **__):  # noqa: D401, ANN001 – stub signature
            print(
                "⚠️  CulturalRetriever disabled – optional dependencies could not be loaded."
            )
            self.is_loaded: bool = False
            self.index = None

        # ------------------------------------------------------------------
        # No-op / graceful-degradation methods
        # ------------------------------------------------------------------
        def build_index(self, *_, **__):  # noqa: ANN001, D401 – stub signature
            raise RuntimeError(
                "CulturalRetriever is unavailable in this environment (missing deps)."
            )

        def save_index(self, *_, **__):  # noqa: ANN001 – stub
            pass

        def load_index(self, *_, **__):  # noqa: ANN001 – stub
            print("⚠️  Cannot load cultural index – CulturalRetriever unavailable.")

        def query(self, *_, **__):  # noqa: ANN001 – stub
            return []

        def query_ctu(self, *_, **__):  # noqa: ANN001 – stub
            return []

        def batch_query(self, *_, **__):  # noqa: ANN001 – stub
            return []

def create_sample_snippets(output_file: Path) -> None:
    """Create sample cultural snippets for testing."""
    sample_snippets = [
        {
            'text': 'Santali farmers traditionally use organic farming methods.',
            'category': 'farming',
            'region': 'Jharkhand',
            'language': 'Santali'
        },
        {
            'text': 'Tribal communities celebrate harvest festivals with traditional dances.',
            'category': 'festival',
            'region': 'Odisha',
            'language': 'Santali'
        },
        {
            'text': 'Women in tribal villages often work in agricultural fields.',
            'category': 'gender',
            'region': 'West Bengal',
            'language': 'Santali'
        },
        {
            'text': 'Traditional tribal houses are made of mud and bamboo.',
            'category': 'housing',
            'region': 'Jharkhand',
            'language': 'Santali'
        },
        {
            'text': 'Santali language uses Ol-Chiki script for writing.',
            'category': 'language',
            'region': 'Jharkhand',
            'language': 'Santali'
        }
    ]
    
    # Convert to DataFrame and save as TSV
    df = pd.DataFrame(sample_snippets)
    df.to_csv(output_file, sep='\t', index=False)
    print(f"Sample snippets saved to {output_file}")

def main():
    """CLI interface for cultural retriever."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cultural retriever for CTU-FlowRAG")
    parser.add_argument("--snippets", required=True, help="Cultural snippets TSV file")
    parser.add_argument("--index-dir", required=True, help="Directory to save/load index")
    parser.add_argument("--query", help="Query text for testing")
    parser.add_argument("--build", action="store_true", help="Build new index")
    parser.add_argument("--create-sample", help="Create sample snippets file")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_snippets(Path(args.create_sample))
        return
    
    # Initialize retriever
    index_path = Path(args.index_dir) / "cultural_index.faiss"
    retriever = CulturalRetriever(index_path if index_path.exists() else None)
    
    if args.build:
        # Build new index
        retriever.build_index(Path(args.snippets), Path(args.index_dir))
    elif args.query:
        # Test query
        if not retriever.is_loaded:
            retriever.load_index(Path(args.index_dir))
        
        results = retriever.query(args.query)
        print(f"\nQuery: {args.query}")
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['snippet']} (sim: {result['similarity']:.3f})")

if __name__ == "__main__":
    main() 