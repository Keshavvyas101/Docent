# Chunking and Embedding in Docent

## How Text Chunking Works

Docent uses LangChain's RecursiveCharacterTextSplitter to break documents into manageable pieces. This splitter tries to split text along natural boundaries like paragraphs, sentences, and words, preserving semantic coherence within each chunk.

### Chunking Parameters

- **chunk_size**: 500 characters. This is the maximum size of each chunk. Smaller chunks provide more precise retrieval but may lose context. Larger chunks preserve more context but reduce retrieval precision.
- **chunk_overlap**: 50 characters. Adjacent chunks overlap by this amount to ensure that information at chunk boundaries is not lost.

### Why These Values?

The chunk size of 500 was chosen as a balance between precision and context. For technical documentation, paragraphs typically contain 200-600 characters, so 500 characters usually captures a complete thought without including too much unrelated information.

The overlap of 50 characters ensures that sentences split across chunk boundaries are captured in at least one chunk. This prevents information loss at the edges.

## How Embeddings Work

Each text chunk is converted to a 384-dimensional dense vector using the all-MiniLM-L6-v2 model from sentence-transformers. This model was chosen for several reasons:

1. **Speed**: It processes text quickly on CPU, making it suitable for local development without a GPU.
2. **Quality**: Despite its small size (80 MB), it produces high-quality embeddings that capture semantic meaning effectively.
3. **Dimension**: The 384-dimensional output is compact enough for efficient storage and retrieval while maintaining good semantic resolution.

### Embedding Process

1. The text chunk is tokenized into subword tokens.
2. Tokens are processed through the transformer model.
3. The output hidden states are mean-pooled to produce a single 384-dimensional vector.
4. The vector is normalized to unit length for cosine similarity computation.

### Similarity Search

When a user asks a question, the question is embedded using the same model and compared against all stored chunk embeddings using cosine similarity. The top-k most similar chunks are retrieved and passed to the language model as context.

Cosine similarity measures the angle between two vectors:
- 1.0 means the vectors point in the same direction (identical meaning)
- 0.0 means the vectors are orthogonal (unrelated)
- -1.0 means opposite directions (opposite meaning)

In practice, most document chunks have similarity scores between 0.2 and 0.8 for related queries.
