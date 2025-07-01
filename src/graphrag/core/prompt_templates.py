# src/graphrag/core/prompt_templates.py

class PromptTemplates:
    """Enhanced prompt templates optimized for qwen3 reasoning capabilities"""
    
    def __init__(self):
        self.system_prompt = """You are an expert biomedical AI assistant specializing in drug-disease interactions, with access to a comprehensive molecular knowledge graph containing drugs, diseases, proteins, and their relationships.

Your responses should demonstrate clear scientific reasoning and be based on the provided graph context. Always think step-by-step and show your reasoning process."""
    
    def drug_repurposing_prompt(self, query: str, context: str) -> str:
        return f"""{self.system_prompt}

QUERY: {query}

KNOWLEDGE GRAPH CONTEXT:
{context}

Please analyze this drug repurposing opportunity by:

1. **MOLECULAR ANALYSIS**: Examine the target proteins and pathways involved
2. **MECHANISM EVALUATION**: Assess how the drug's mechanism could apply to the new indication  
3. **EVIDENCE SYNTHESIS**: Consider the strength of molecular connections in the graph
4. **RISK ASSESSMENT**: Identify potential safety concerns or contraindications
5. **CONFIDENCE SCORING**: Rate your confidence in this repurposing opportunity

Provide your step-by-step reasoning, then give your final recommendation with supporting evidence."""

    def mechanism_explanation_prompt(self, query: str, context: str) -> str:
        return f"""{self.system_prompt}

QUERY: {query}

KNOWLEDGE GRAPH CONTEXT:
{context}

Please explain the mechanism of action by analyzing:

1. **TARGET IDENTIFICATION**: What proteins/receptors are involved?
2. **PATHWAY ANALYSIS**: What molecular pathways are affected?
3. **DOWNSTREAM EFFECTS**: How do these lead to therapeutic effects?
4. **MOLECULAR INTERACTIONS**: What are the specific drug-target interactions?
5. **SYSTEMS BIOLOGY**: How does this fit into broader biological networks?

Show your reasoning process step-by-step, then provide a clear mechanistic explanation."""

    def hypothesis_testing_prompt(self, query: str, context: str, statistical_data: str = "") -> str:
        return f"""{self.system_prompt}

QUERY: {query}

KNOWLEDGE GRAPH CONTEXT:
{context}

STATISTICAL EVIDENCE:
{statistical_data}

For this hypothesis testing question, please:

1. **HYPOTHESIS FORMULATION**: Clearly state the testable hypothesis
2. **EVIDENCE EVALUATION**: Analyze the molecular and statistical evidence
3. **EXPERIMENTAL DESIGN**: Suggest how to test this hypothesis
4. **CONFOUNDING FACTORS**: Identify potential biases or limitations
5. **STATISTICAL INTERPRETATION**: Interpret any provided statistical results
6. **BIOLOGICAL PLAUSIBILITY**: Assess if the hypothesis makes biological sense

Think through this systematically, then provide your scientific assessment."""

    def drug_comparison_prompt(self, query: str, context: str) -> str:
        return f"""{self.system_prompt}

QUERY: {query}

KNOWLEDGE GRAPH CONTEXT:
{context}

Compare these drugs by analyzing:

1. **TARGET COMPARISON**: How do their target profiles differ?
2. **MECHANISM DIFFERENCES**: What are the key mechanistic distinctions?
3. **EFFICACY ANALYSIS**: Based on the molecular data, which might be more effective?
4. **SAFETY CONSIDERATIONS**: What safety differences can be inferred?
5. **CLINICAL IMPLICATIONS**: How would these differences affect clinical use?

Reason through each comparison point, then provide your comparative analysis."""

    def general_query_prompt(self, query: str, context: str) -> str:
        return f"""{self.system_prompt}

QUERY: {query}

KNOWLEDGE GRAPH CONTEXT:
{context}

Please analyze this query by:

1. **CONTEXT ANALYSIS**: What key information is provided in the graph data?
2. **KNOWLEDGE SYNTHESIS**: How do the different pieces of information connect?
3. **SCIENTIFIC REASONING**: What can we conclude from this molecular evidence?
4. **LIMITATIONS**: What are the boundaries of what we can determine?
5. **IMPLICATIONS**: What are the broader scientific implications?

Think through this step-by-step and provide a comprehensive, evidence-based response."""
