# Glossary

**attention score** — The query–key comparison becomes an attention score through a dot product.  
*Introduced in sec_01_01.*

**information bottleneck** — This creates an information bottleneck: the hidden state has finite capacity, while the sequence may contain many distinct facts, references, and relationships.  
*Introduced in sec_01_01.*

**key** — Whereas the query represents what the current position is looking for, the key represents what a position makes available for matching.  
*Introduced in sec_01_01.*

**key vector** — Each position also produces a key vector.  
*Introduced in sec_01_01.*

**query** — It encodes search criteria in a learned vector space.  
*Introduced in sec_01_01.*

**query vector** — For each token position, attention constructs a query vector: a numeric representation of what that position needs to find in the surrounding sequence.  
*Introduced in sec_01_01.*

**scaled attention score** — The dot product must be scaled before it is used.  
*Introduced in sec_01_01.*

**value** — The value is the content that position contributes if attention selects it.  
*Introduced in sec_01_01.*

**value vector** — Each position produces a value vector through another learned projection.  
*Introduced in sec_01_01.*
