# Attention and Retrieval Systems


# 1. Sequence Modeling Limitations

> This chapter built an understanding of how recurrent neural networks process sequences through repeated updates to a fixed-size hidden state. It showed that this state compresses the preceding prefix, requiring each new token to overwrite or merge information into the same representation. By the end, the reader can explain why long sequences weaken or obscure earlier details, identify the resulting long-range dependency problem, and motivate architectures that directly access relevant earlier representations.

## Sequence Compression into a Recurrent State

An RNN processes a sequence in order, one token at a time. Each token is first represented as a vector, then combined with the model’s current hidden state. The result becomes the hidden state for the next position:

\[
h_t = f(x_t, h_{t-1})
\]

Here, \(x_t\) represents the token at position \(t\), \(h_{t-1}\) is the summary carried from the previous position, and \(h_t\) is the updated summary after reading the current token. The function \(f\) is learned; in a basic RNN, it applies a transformation to both inputs and passes the result through a nonlinear activation.

Consider the sequence “the customer cannot access the account.” The model begins with an initial state, often a vector of zeros. After reading “the,” it updates that state. It then reads “customer” using the state produced by “the,” updates it again, and continues until it reaches “account.” At every position, the hidden state contains the information the network has chosen to carry forward from the prefix seen so far.

A downstream component can use the state at position \(t\) to make a prediction based on the sequence through that position. The final state therefore acts as a compact representation of everything the RNN has processed. Information does not travel independently alongside the sequence; it is incorporated into this evolving vector as each new token arrives.

The constraint becomes clearer as the sequence grows. The hidden state has a fixed size, regardless of whether the model has read five tokens or five hundred. Every new token must be incorporated into that same vector:

\[
h_t = f(x_t, h_{t-1})
\]

The model cannot preserve an independent representation of every earlier token in a separate channel. Instead, it must update the existing summary so that it reflects both the old context and the new input. If the new token carries information that conflicts with, or is more useful than, something already represented, the update may alter the dimensions that carried the earlier information. Some details are weakened; others may be overwritten entirely.

Return to “the customer cannot access the account.” After processing “the customer,” the state may encode that the sequence concerns a customer. As “cannot access” arrives, it must also represent the problem and its relationship to the customer. By the time the model processes “account,” the same fixed-capacity state must contain enough information about the subject, the failure, the relevant relationships, and the latest token.

This is not a matter of the model failing to recognize a token. The token can be read correctly when it arrives. The difficulty is preserving information from that token through every subsequent update. Each additional position creates another opportunity for an earlier detail to be diluted by the accumulated context. The final state is therefore a compressed summary, and compression necessarily risks discarding distinctions that a later prediction might need.

The loss of long-range information is easiest to see when a later prediction depends on a detail introduced near the beginning. In “the customer cannot access the account,” a prediction at “account” may need to preserve the subject established by “customer” and the failure expressed by “cannot access.” Those details arrive before the final token, yet they must survive each subsequent update.

With only a few intervening tokens, the state may retain enough of both. As the distance grows, however, preservation becomes more difficult. Each update transforms the previous state together with the new token. The representation of an early detail is therefore repeatedly mixed with later information, rather than carried forward as an untouched record. A detail can remain partially represented, but its exact relationship to the later context may become indistinguishable from other possibilities.

This creates a practical failure mode. Two sequences can differ in an early token that matters to the final prediction, while their hidden states near the end become sufficiently similar that a downstream component cannot reliably tell them apart. The model has not necessarily misread the early token. It may have represented it correctly at the position where it appeared, then lost the distinction during later updates.

The problem is especially severe when the relevant dependency spans many positions. Information needed at the end must pass through every intermediate recurrent transition, and each transition can weaken or overwrite it. By the final position, the state may still summarize the sequence plausibly while no longer preserving the specific early fact that the prediction requires.

The bottleneck therefore defines a requirement for the next architecture. A prediction at one position should not depend exclusively on information compressed into the single state produced by every preceding position. It should have a path to the earlier representations that may contain the detail it needs.

For the sequence “the customer cannot access the account,” a representation associated with “account” should be able to use the earlier representation of “customer” and the phrase “cannot access” without requiring those details to survive perfectly in one repeatedly updated summary. The model still needs to combine context, but the combination should not force every dependency through the same narrow route.

This requirement changes the question the model asks of its context. Instead of asking, “What summary of the entire prefix can I carry forward?” the model can ask, “Which earlier positions are relevant to this current position?” A position near the end can then draw more strongly on the parts of the sequence that determine its interpretation and less strongly on unrelated material.

The design must also support different relationships at the same time. One prediction may depend on a nearby syntactic connection; another may require a reference established much earlier. A useful architecture should represent both without assuming that every dependency has the same distance or importance.

The next mechanism addresses this by allowing positions to compare their needs with information held at other positions. It replaces exclusive reliance on one evolving recurrent summary with direct, learned access across the sequence.


# 2. Self-Attention Mechanics

## Constructing a Query Vector

*[NOT YET WRITTEN — sec_02_01]*

## Constructing a Key Vector

*[NOT YET WRITTEN — sec_02_02]*

## Constructing a Value Vector

*[NOT YET WRITTEN — sec_02_03]*

## Applying Dimension-Based Score Scaling

*[NOT YET WRITTEN — sec_02_04]*

## Defining the Scaled Attention Operation

*[NOT YET WRITTEN — sec_02_05]*

## Applying Softmax to Attention Scores

*[NOT YET WRITTEN — sec_02_06]*

## Summing Retrieved Content

*[NOT YET WRITTEN — sec_02_07]*

## Understanding Self-Attention as a Complete Pipeline

*[NOT YET WRITTEN — sec_02_08]*


# 3. Positional Representation

## The Problem of Permutation Invariance

*[NOT YET WRITTEN — sec_03_01]*

## Learned Positional Embedding

*[NOT YET WRITTEN — sec_03_02]*

## Sinusoidal Positional Encoding

*[NOT YET WRITTEN — sec_03_03]*

## Rotary Position Embedding

*[NOT YET WRITTEN — sec_03_04]*


# 4. Multi-Head Attention and Projections

## Why Multiple Attention Heads Are Needed

*[NOT YET WRITTEN — sec_04_01]*

## Learned Query, Key, and Value Transformations

*[NOT YET WRITTEN — sec_04_02]*

## Collecting the Outputs of Attention Heads

*[NOT YET WRITTEN — sec_04_03]*

## The Role of the Output Transformation

*[NOT YET WRITTEN — sec_04_04]*


# 5. Attention Inference and Key-Value Caching

## Stored Attention States for Future Tokens

*[NOT YET WRITTEN — sec_05_01]*

## Sequence-Length-Dependent Memory Scaling

*[NOT YET WRITTEN — sec_05_02]*


# 6. Retrieval-Augmented Generation Architecture

## Defining the External Knowledge Corpus

*[NOT YET WRITTEN — sec_06_01]*

## Generating Answers Grounded in Retrieved Evidence

*[NOT YET WRITTEN — sec_06_02]*


# 7. Document Chunking and Context Design

## Establishing the 1,500-2,000-Token Chunk Target

*[NOT YET WRITTEN — sec_07_01]*

## Preventing Context Loss in Small Chunks

*[NOT YET WRITTEN — sec_07_02]*

## Managing Semantic Dilution in Large Chunks

*[NOT YET WRITTEN — sec_07_03]*

## Designing Self-Contained Retrieval Units

*[NOT YET WRITTEN — sec_07_04]*

## Using Document Structure as a Chunking Signal

*[NOT YET WRITTEN — sec_07_05]*

## Applying Zero-Overlap Chunking

*[NOT YET WRITTEN — sec_07_06]*


# 8. Dense Embeddings and Vector Retrieval

## Choosing and Using an Embedding Model

*[NOT YET WRITTEN — sec_08_01]*

## Constructing Dense Vector Embeddings

*[NOT YET WRITTEN — sec_08_02]*

## Measuring Semantic Vector Similarity

*[NOT YET WRITTEN — sec_08_03]*

## Independent Embedding Retrieval

*[NOT YET WRITTEN — sec_08_04]*

## Storing Embeddings in a Vector Database

*[NOT YET WRITTEN — sec_08_05]*

## Approximate Nearest-Neighbor Search

*[NOT YET WRITTEN — sec_08_06]*

## Selecting Top-K Candidate Passages

*[NOT YET WRITTEN — sec_08_07]*


# 9. Retrieval Provenance and Metadata

## Identifying Retrieved Documents

*[NOT YET WRITTEN — sec_09_01]*

## Recording Where Content Came From

*[NOT YET WRITTEN — sec_09_02]*

## Classifying Retrieved Sources

*[NOT YET WRITTEN — sec_09_03]*

## Preserving Temporal Context

*[NOT YET WRITTEN — sec_09_04]*


# 10. Candidate Reranking

## The Role of a Reranker After Initial Retrieval

*[NOT YET WRITTEN — sec_10_01]*

## Pairwise Query-Passage Relevance Scoring

*[NOT YET WRITTEN — sec_10_02]*
