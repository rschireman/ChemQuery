# ChemQuery
## RAG demo running gemma-7b-it completely locally on an RTX 3070 (or any gpu with at least 8 Gb VRAM)

This notebook demonstrates how to run the Retrieval-Augmented Generation (RAG) model using the Gemma-7B-IT model entirely on a local machine equipped with an NVIDIA RTX 3070 GPU. Specifically, we will ask Gemma about the CRYSTAL23 manual (https://www.crystal.unito.it/).

The primary goal of Retrieval-Augmented Generation (RAG) is to enhance the output quality of Large Language Models (LLMs).

## Key Improvements
1. Preventing Hallucinations
LLMs are powerful, but they can sometimes generate information that appears correct but is actually inaccurate — a phenomenon known as hallucination. RAG addresses this by incorporating factual data through a retrieval process, ensuring more accurate outputs. Additionally, if a generated response seems incorrect, the retrieval process provides access to the original sources, enabling verification of the information.

2. Utilizing Custom Data
Many LLMs are trained on extensive internet-scale text data, giving them excellent language modeling capabilities but often lacking in specific domain knowledge (like computational chemistry). RAG systems enhance LLMs by integrating domain-specific data, such as computational chemistry software documentation, allowing the models to produce tailored outputs for specialized use cases.

## Why run locally?

LLMs are incredibly resource-intensive due to their size and complexity (Gemma-7b has 7 _billion_ parameters. They require significant computational power and memory, often necessitating specialized hardware like high-end GPUs. We will use quantization (for a more detailed description, see the notebook) to fit this model into the 8 Gb of VRAM of a relatively old GPU. While running LLMs locally presents various challenges, it also offers substantial benefits:

1. Data Privacy and Security:
Running LLMs locally ensures that sensitive data remains on-premises, reducing the risk of data breaches and ensuring compliance with privacy regulations.

2. Cost Efficiency:
For organizations with high usage requirements, running LLMs locally can be more cost-effective over time compared to the recurring costs of cloud-based services.
