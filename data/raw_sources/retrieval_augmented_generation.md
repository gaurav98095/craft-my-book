# Lecture 7 — Retrieval Augmented Generation

## The problem RAG solves

A language model's knowledge is frozen at training time and bounded by what
fit in its context window. RAG (retrieval augmented generation) sidesteps
both limits: instead of asking the model to recall a fact from its weights,
you first retrieve the passages most likely to contain that fact from an
external corpus, then hand those passages to the model as context and ask it
to answer using them. The model still does the reasoning; it no longer has
to do the remembering.

## The three moving parts

**Chunking.** A document is split into pieces small enough to retrieve
individually and cheap enough to embed, but large enough to still make sense
out of context. Splitting mid-sentence or mid-table produces fragments that
retrieve well by keyword overlap but read as nonsense once retrieved — chunk
boundaries should respect the document's own structure wherever the format
gives you one (paragraphs, slide boundaries, table rows).

**Embedding and the vector database.** Each chunk is converted into a dense
vector by an embedding model, such that semantically similar chunks land
near each other in vector space. Those vectors are stored in a vector
database (Chroma, in this pipeline) that supports approximate nearest
neighbour search: given a query vector, return the k chunks whose vectors
are closest to it, without scanning every chunk in the corpus.

**Reranking.** Nearest-neighbour retrieval over embeddings is fast but
approximate — it optimizes for semantic similarity, not necessarily for "is
this the passage that answers the question." A reranker takes the top-k
candidates from the vector search and re-scores them with a more expensive,
more accurate model (often a cross-encoder that looks at the query and each
candidate together, rather than embedding them independently), then returns
a shorter, better-ordered list.

## Why chunk size is a real design decision, not a detail

Too small, and a chunk loses the context it needs to be self-contained — a
paragraph that says "this approach also fails" means nothing without the
paragraph before it that named the approach. Too large, and the embedding
for the chunk becomes an average over several unrelated ideas, which hurts
retrieval precision: a query about one specific idea in a five-idea chunk
retrieves the whole chunk, diluted. A useful default is to target
1,500–2,000 tokens per chunk, with no overlap between chunks — overlap
sounds like it should help, but it means the same sentence gets embedded
(and can get retrieved) from two different chunks, double-counting it in any
downstream analysis that assumes chunks partition the corpus.

## Provenance

A retrieved chunk is only as trustworthy as its source. A production RAG
system should carry, alongside every chunk, where it came from: which
document, what type of source (a slide deck reads differently from a
transcript), and — for anything derived from a recording — a timestamp. That
provenance is what lets a downstream system say "as the lecture demonstrated
on the whiteboard at 34 minutes in" instead of presenting every fact with
the same flat authority regardless of where it came from.

## Where this connects back

The chunking and embedding steps described here are exactly what Stage 4 and
Stage 5 of this pipeline's own ingestion process do to its own source
material — this pipeline is, among other things, a RAG system whose corpus
happens to be the lecture material it was fed.
