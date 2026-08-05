# Lecture 3 — Attention Mechanisms

## Why attention

Recurrent networks read a sequence one token at a time and compress everything
they've seen into a single hidden state. That hidden state is a bottleneck:
by the time an RNN reaches token 200, whatever it learned about token 3 has
usually been overwritten many times over. Attention removes the bottleneck by
letting every output position look directly at every input position, and
decide for itself how much each one matters.

## Self-attention, mechanically

Given a sequence of token embeddings, self-attention computes three vectors
per token: a **query**, a **key**, and a **value**. The query is "what this
token is looking for"; the key is "what this token has to offer"; the value
is "what this token actually contributes if selected."

For each token, the attention score against every other token is the dot
product of its query with that token's key, scaled by the square root of the
key dimension:

```
score(i, j) = (Q_i · K_j) / sqrt(d_k)
```

Those scores are passed through a softmax so they sum to 1, and the token's
new representation is the weighted sum of every value vector, weighted by
that softmax distribution. A token that is highly relevant to the query gets
a large weight; an irrelevant one gets pushed toward zero.

## Multi-head attention

A single attention computation learns one notion of "relevant." Multi-head
attention runs several of these in parallel, each with its own learned Q/K/V
projections, then concatenates the results and projects back down. In
practice one head often specializes in short-range syntactic relationships
(a verb attending to its subject) while another specializes in long-range
coreference (a pronoun attending to the noun it refers to), though nothing in
the architecture forces this division — it emerges from training.

## Positional encoding

Attention itself is permutation-invariant: shuffle the input tokens and the
attention scores between any two specific tokens don't change. That's a
problem for language, where order carries meaning ("dog bites man" is not
"man bites dog"). Positional encoding injects order back in, either as a
fixed sinusoidal pattern added to each token's embedding, or as a learned
embedding per position. Rotary position embeddings (RoPE), used in most
current large language models, take a third approach: they rotate the query
and key vectors by an angle proportional to position, so the dot product
between two tokens naturally encodes their relative distance.

## The KV cache

At inference time, generating one token at a time, recomputing every key and
value vector for the whole sequence at every step would be wasteful — the
keys and values for tokens already generated never change. The KV cache
stores them once and reuses them, so each new token only requires computing
its own query, key, and value, then attending over the cached keys and
values of everything before it. This is why long-context generation is
memory-bound rather than compute-bound: the KV cache grows linearly with
sequence length, and at long enough contexts it dominates GPU memory.

## What this sets up

Everything above describes one transformer block's attention sub-layer. The
next lecture covers what surrounds it: the feed-forward sub-layer, residual
connections, and layer normalization — and why removing any one of those
three makes deep transformers stop training.
