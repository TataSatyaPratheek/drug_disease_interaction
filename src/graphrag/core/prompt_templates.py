# src/graphrag/core/prompt_templates.py

class PromptTemplates:
    """Domain-specific prompt templates for drug-disease GraphRAG"""
    
    def __init__(self):
        self.system_prompt = """You are a biomedical AI assistant specializing in drug-disease interactions. 
You have access to a comprehensive knowledge graph containing drugs, diseases, proteins, and their relationships.
Always provide accurate, scientific responses based on the provided graph context.
When you don't have enough information, clearly state the limitations."""
    
    def drug_repurposing_prompt(self, query: str, context: str) -> str:
        return f"""{self.system_prompt}

QUERY: {query}

KNOWLEDGE GRAPH CONTEXT:
{context}

Based on the knowledge graph data above, provide a comprehensive analysis of drug repurposing opportunities. Include:
1. List of candidate drugs with their mechanisms
2. Scientific rationale for each candidate
3. Potential risks or considerations
4. Confidence level of each recommendation

Response:"""
    
    def mechanism_explanation_prompt(self, query: str, context: str) -> str:
        return f"""{self.system_prompt}

QUERY: {query}

KNOWLEDGE GRAPH CONTEXT:
{context}

Based on the molecular pathways and relationships shown above, explain:
1. The step-by-step mechanism of action
2. Key proteins and their roles
3. How the drug affects the disease pathway
4. Any alternative or related mechanisms

Response:"""
    
    def drug_comparison_prompt(self, query: str, context: str) -> str:
        return f"""{self.system_prompt}

QUERY: {query}

KNOWLEDGE GRAPH CONTEXT:
{context}

Compare the drugs based on the graph data above. Include:
1. Similarities in targets and mechanisms
2. Key differences in action
3. Relative advantages/disadvantages
4. Clinical implications of the differences

Response:"""
    
    def target_discovery_prompt(self, query: str, context: str) -> str:
        return f"""{self.system_prompt}

QUERY: {query}

KNOWLEDGE GRAPH CONTEXT:
{context}

Based on the protein and pathway information above, provide:
1. Key target proteins and their roles
2. Druggability assessment
3. Potential therapeutic approaches
4. Current drugs targeting these proteins

Response:"""
    
    def general_query_prompt(self, query: str, context: str) -> str:
        return f"""{self.system_prompt}

QUERY: {query}

KNOWLEDGE GRAPH CONTEXT:
{context}

Based on the knowledge graph information above, provide a comprehensive answer to the query.
Focus on factual information from the graph and clearly indicate any limitations.

Response:"""
    
    def hypothesis_testing_prompt(self, query: str, context: str, statistical_data: str = "") -> str:
        return f"""{self.system_prompt}

QUERY: {query}

KNOWLEDGE GRAPH CONTEXT:
{context}

STATISTICAL ANALYSIS:
{statistical_data}

You are helping with biomedical hypothesis testing. Based on the graph context and any statistical data provided:
1. Formulate testable hypotheses
2. Suggest experimental approaches
3. Identify required controls
4. Predict expected outcomes
5. Discuss potential confounding factors

Response:"""
