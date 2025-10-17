#!/usr/bin/env python
"""Tutorial: Using PromptAgent for Structured Extraction

This tutorial demonstrates how to use the new PromptAgent class to extract
structured information from scientific papers using the prompt registry.

The PromptAgent provides a high-level interface that:
1. Automatically loads prompts from PROMPT_REGISTRY
2. Executes the full RAG pipeline (chunking → retrieval → generation)
3. Parses and validates LLM output for structured prompts
4. Returns typed results with error handling
"""

from nerxiv.agents.prompt_agent import PromptAgent

# Sample scientific text about a material
SAMPLE_TEXT = """
We investigate the electronic structure of strontium vanadate (SrVO3) using 
density functional theory combined with dynamical mean-field theory (DFT+DMFT). 
SrVO3 is a correlated metal with a partially filled t2g band. The system shows 
strong correlation effects that are captured well by the DMFT approach.

Our calculations show that the quasiparticle peak at the Fermi level is well 
described by the DFT+DMFT method. We use the continuous-time quantum Monte Carlo 
(CT-QMC) impurity solver to solve the DMFT equations. The results are in good 
agreement with photoemission spectroscopy experiments.
"""


def example_basic_usage():
    """Example 1: Basic usage with material formula extraction."""
    print("=" * 80)
    print("Example 1: Extract material formula (unstructured)")
    print("=" * 80)
    
    # Create agent for material formula extraction
    agent = PromptAgent(
        query_name="material_formula",
        chunker="Chunker",
        retriever_model="all-MiniLM-L6-v2",
        llm_model="deepseek-r1",
        temperature=0.2,  # Low temperature for consistent output
    )
    
    # Run extraction
    result = agent.run(text=SAMPLE_TEXT, n_top_chunks=3)
    
    # Display results
    print(f"\nRetrieved chunks ({len(result['chunks'])} total):")
    print(result["retrieved"][:200] + "...")
    
    print(f"\nLLM Answer:")
    print(result["answer"])
    
    print(f"\nParsed result: {result['parsed']}")
    print()


def example_structured_extraction():
    """Example 2: Structured extraction with schema validation."""
    print("=" * 80)
    print("Example 2: Extract material formula (structured)")
    print("=" * 80)
    
    # Create agent for structured material formula extraction
    agent = PromptAgent(
        query_name="material_formula_structured",
        chunker="SemanticChunker",  # Use semantic chunking
        retriever_model="all-MiniLM-L6-v2",
        llm_model="deepseek-r1",
        temperature=0.2,
        format="json",  # Request JSON format from LLM
    )
    
    # Run extraction
    result = agent.run(text=SAMPLE_TEXT, n_top_chunks=5)
    
    # Display results
    print(f"\nNumber of chunks: {len(result['chunks'])}")
    print(f"\nPrompt sent to LLM:")
    print(result["prompt"][:300] + "...")
    
    print(f"\nRaw LLM Answer:")
    print(result["answer"])
    
    # The parsed field contains validated Pydantic model data
    if result["parsed"]:
        print(f"\n✓ Parsed and validated result:")
        print(f"  IUPAC formula: {result['parsed'].get('iupac')}")
        print(f"  Anonymous formula: {result['parsed'].get('anonymous')}")
        print(f"  Hill formula: {result['parsed'].get('hill')}")
        print(f"  Reduced formula: {result['parsed'].get('reduced')}")
    else:
        print(f"\n✗ Parsing failed: {result.get('parse_error')}")
    print()


def example_dmft_check():
    """Example 3: Check if DMFT method is used (boolean query)."""
    print("=" * 80)
    print("Example 3: Check if DMFT is used")
    print("=" * 80)
    
    # Create agent for DMFT detection
    agent = PromptAgent(
        query_name="only_dmft",
        chunker="Chunker",
        retriever_model="all-MiniLM-L6-v2",
        llm_model="deepseek-r1",
        temperature=0.1,  # Very low for binary classification
    )
    
    # Run extraction
    result = agent.run(text=SAMPLE_TEXT, n_top_chunks=5)
    
    # Display results
    print(f"\nLLM Answer: {result['answer']}")
    print(f"\nDMFT is used: {result['answer'].strip() == 'True'}")
    print()


def example_error_handling():
    """Example 4: Error handling with invalid inputs."""
    print("=" * 80)
    print("Example 4: Error handling")
    print("=" * 80)
    
    # Test 1: Invalid query name
    try:
        agent = PromptAgent(query_name="nonexistent_query")
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    # Test 2: Invalid chunker
    try:
        agent = PromptAgent(
            query_name="material_formula",
            chunker="InvalidChunker"
        )
        print("✗ Should have raised KeyError")
    except KeyError as e:
        print(f"✓ Caught expected error: {e}")
    
    # Test 3: Empty text
    try:
        agent = PromptAgent(query_name="material_formula")
        result = agent.run(text="")
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    print()


def example_custom_llm_options():
    """Example 5: Using custom LLM options."""
    print("=" * 80)
    print("Example 5: Custom LLM options")
    print("=" * 80)
    
    # Create agent with custom LLM parameters
    agent = PromptAgent(
        query_name="material_formula",
        chunker="AdvancedSemanticChunker",
        retriever_model="all-MiniLM-L6-v2",
        llm_model="llama3.1:70b",
        # Custom LLM options
        temperature=0.3,
        top_p=0.9,
        top_k=40,
        num_ctx=8192,
        format="json",
    )
    
    print(f"Agent configured with:")
    print(f"  - Chunker: AdvancedSemanticChunker")
    print(f"  - Retriever: all-MiniLM-L6-v2")
    print(f"  - LLM: llama3.1:70b")
    print(f"  - Temperature: 0.3")
    print(f"  - Context window: 8192")
    print()


if __name__ == "__main__":
    print("\nPromptAgent Tutorial")
    print("=" * 80)
    print("This tutorial shows how to use PromptAgent for extraction tasks.")
    print()
    
    # Note: These examples require optional dependencies to be installed
    # Run: pip install -e . to install all dependencies
    
    try:
        example_basic_usage()
        example_structured_extraction()
        example_dmft_check()
        example_error_handling()
        example_custom_llm_options()
        
        print("=" * 80)
        print("Tutorial completed!")
        print("=" * 80)
        
    except ImportError as e:
        print(f"⚠ Warning: Optional dependencies not installed: {e}")
        print("Install with: pip install -e .")
        print("\nShowing error handling example only:")
        example_error_handling()
