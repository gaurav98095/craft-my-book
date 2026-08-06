# Attention and Retrieval: Foundations of Modern Language Models


# 1. Attention Representations and Information Flow

## Query Vector

A recurrent network processes a sequence one token at a time. After reading each token, it updates a single hidden state that must carry everything relevant from the prefix it has seen. This creates an information bottleneck: the hidden state has finite capacity, while the sequence may contain many distinct facts, references, and relationships.

Consider a token near the end of a long customer message processed by SupportBot. Its interpretation might depend on a product name near the beginning, a negation several words earlier, or a condition introduced in the previous paragraph. In a recurrent network, all of that information must survive through every intermediate hidden-state update. By the time the network reaches the final token, details from the beginning may have been weakened or overwritten.

Attention changes the access pattern. Rather than requiring the current position to recover the entire past from one compressed state, it allows each output position to inspect the input sequence directly. A token can consider information from nearby positions and from distant positions in the same computation. The representation produced for that position is therefore based on a selective view of the sequence, not solely on what survived in a running summary.

This does not mean that every token contributes equally. The mechanism must determine which positions are relevant to the token currently being represented. That determination begins with a token expressing what information it needs, then comparing that need with what other tokens offer. The next question is how to represent that need and perform the comparison without confusing the act of searching with the information ultimately returned.

For each token position, attention constructs a query vector: a numeric representation of what that position needs to find in the surrounding sequence. The query is not a natural-language question, and it does not contain the answer. It encodes search criteria in a learned vector space.

Suppose a SupportBot message contains a later phrase such as “that charge.” The representation at this position may need information that identifies which earlier transaction the phrase refers to. Its query can therefore encode features associated with reference resolution, such as the need for a compatible noun phrase or a preceding transaction description. A different position, such as a verb, may produce a query that seeks its subject or an object. The same sequence supplies the inputs, but every position can ask for a different kind of information.

The model produces these queries by applying a learned linear projection to each token representation:

\[
Q_i = X_i W_Q
\]

Here, \(X_i\) is the representation at position \(i\), \(W_Q\) is a learned parameter matrix, and \(Q_i\) is the resulting query vector. The projection gives the model a space in which useful relationships can be expressed. During training, the parameters are adjusted so that queries associated with a position become compatible with representations at positions that can help interpret it.

A query therefore belongs to the position doing the looking. It describes the information that position is seeking, rather than the information another position will ultimately provide. To use that request, the model must compare it with every position in the sequence.

Each position also produces a key vector. Whereas the query represents what the current position is looking for, the key represents what a position makes available for matching. The model applies another learned projection to the same input representation:

\[
K_j = X_j W_K
\]

For a query at position \(i\), attention compares \(Q_i\) with the key at every position \(j\) in the sequence. The comparison asks whether the information sought at position \(i\) is compatible with what position \(j\) offers. This includes nearby positions, distant positions, and—unless the architecture applies a causal restriction—the position itself.

Return to the SupportBot phrase “that charge.” Its query is compared with the keys produced by earlier words and by the rest of the message. A key associated with a transaction description may match strongly because it offers the kind of information needed to resolve the reference. Keys associated with unrelated details may match weakly. At this point, the model has identified degrees of relevance; it has not yet combined the selected information.

That distinction depends on a third vector: the value. Each position produces a value vector through another learned projection:

\[
V_j = X_j W_V
\]

The value is the content that position contributes if attention selects it. A key helps determine whether a position should be consulted; its corresponding value supplies what is read from that position. Thus, the key for a transaction description might make the position easy to select, while the value carries the representation of the transaction details used to interpret “that charge.”

The query belongs to the reader, the key describes a possible match, and the value carries the possible contribution. Attention keeps these roles separate while evaluating every position.

The query–key comparison becomes an attention score through a dot product. For a query at position \(i\) and a key at position \(j\), the unnormalized score is

\[
s_{ij} = Q_i \cdot K_j
\]

A large dot product indicates that the features sought by position \(i\) align with the features offered by position \(j\). A small or negative dot product indicates a weaker match. The model computes this score between the query and every key available to that position.

The dot product must be scaled before it is used:

\[
\operatorname{score}(i,j)
=
\frac{Q_i \cdot K_j}{\sqrt{d_k}}
\]

Here, \(d_k\) is the dimensionality of each key vector. As the key dimension grows, an unscaled dot product tends to produce values with larger magnitude. Those large values can make the subsequent distribution overly concentrated, giving training less useful gradient information. Dividing by \(\sqrt{d_k}\) keeps the scores at a more manageable scale.

For SupportBot, the query associated with “that charge” produces one scaled score for the key at each candidate position. The transaction description may receive a high score, while unrelated positions receive lower scores. These scores express relative compatibility, not final selection. They have not yet been converted into weights, and no value vector has yet been added to the representation at “that charge.”

The next operation normalizes the scores across the candidate positions. The resulting weights determine how strongly each corresponding value contributes to the output representation.

## Key Vector

*[NOT YET WRITTEN — sec_01_02]*

## Value Vector

*[NOT YET WRITTEN — sec_01_03]*


# 2. Scaled Dot-Product Attention

## Why Key Dimensions Affect Score Magnitude

*[NOT YET WRITTEN — sec_02_01]*

## The Complete Scaled Dot-Product Attention Operation

*[NOT YET WRITTEN — sec_02_02]*

## Producing Softmax Attention Weights

*[NOT YET WRITTEN — sec_02_03]*

## Combining Contributions into an Attention Output

*[NOT YET WRITTEN — sec_02_04]*


# 3. Multi-Head Attention Architecture

## The Role of an Attention Head

*[NOT YET WRITTEN — sec_03_01]*

## Query, Key, and Value Representations

*[NOT YET WRITTEN — sec_03_02]*

## Why Multiple Attention Heads Are Useful

*[NOT YET WRITTEN — sec_03_03]*

## Concatenating Parallel Head Results

*[NOT YET WRITTEN — sec_03_04]*

## The Attention Output Projection

*[NOT YET WRITTEN — sec_03_05]*


# 4. Positional Representation

## Permutation Invariance in Self-Attention

*[NOT YET WRITTEN — sec_04_01]*

## Sinusoidal Positional Encoding

*[NOT YET WRITTEN — sec_04_02]*

## Learned Absolute Position Tables

*[NOT YET WRITTEN — sec_04_03]*

## Rotating Attention Representations

*[NOT YET WRITTEN — sec_04_04]*


# 5. Efficient Autoregressive Inference

## Storing Prior Attention States

*[NOT YET WRITTEN — sec_05_01]*

## Linear Memory Scaling with Generated Tokens

*[NOT YET WRITTEN — sec_05_02]*


# 6. Retrieval-Augmented Generation Motivation

## The Limits of Internal Model Representations

*[NOT YET WRITTEN — sec_06_01]*

## Knowledge Bound to the Training Data

*[NOT YET WRITTEN — sec_06_02]*

## Finite Input Capacity as a Design Constraint

*[NOT YET WRITTEN — sec_06_03]*

## Retrieving Evidence at Inference Time

*[NOT YET WRITTEN — sec_06_04]*

## Retrieval-Augmented Generation as the Motivated Solution

*[NOT YET WRITTEN — sec_06_05]*


# 7. Document Preparation and Chunking

## Document Chunking Fundamentals

*[NOT YET WRITTEN — sec_07_01]*

## Structure-Aware Chunk Boundaries

*[NOT YET WRITTEN — sec_07_02]*

## The 1,500–2,000-Token Chunk Target

*[NOT YET WRITTEN — sec_07_03]*

## Zero Chunk Overlap

*[NOT YET WRITTEN — sec_07_04]*


# 8. Embeddings and Vector Indexing

## Choosing an Embedding Model

*[NOT YET WRITTEN — sec_08_01]*

## Semantic Representation of Document Chunks

*[NOT YET WRITTEN — sec_08_02]*

## Storage Requirements for Vector Data

*[NOT YET WRITTEN — sec_08_03]*

## Approximate Nearest-Neighbor Search

*[NOT YET WRITTEN — sec_08_04]*


# 9. Candidate Retrieval and Reranking

## Semantic Similarity Retrieval

*[NOT YET WRITTEN — sec_09_01]*

## Selecting the Initial Candidate Pool

*[NOT YET WRITTEN — sec_09_02]*

## The Purpose of Reranking

*[NOT YET WRITTEN — sec_09_03]*

## Cross-Encoder Reranker Architecture

*[NOT YET WRITTEN — sec_09_04]*

## Answer-Relevance Scoring

*[NOT YET WRITTEN — sec_09_05]*


# 10. Retrieval Provenance and Source Metadata

## Core Provenance Metadata

*[NOT YET WRITTEN — sec_10_01]*

## Recording When Evidence Was Retrieved

*[NOT YET WRITTEN — sec_10_02]*

## Classifying Retrieved Sources

*[NOT YET WRITTEN — sec_10_03]*
