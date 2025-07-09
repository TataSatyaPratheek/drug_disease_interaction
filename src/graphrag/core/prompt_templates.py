class ScientificPromptTemplates:
    """Enhanced prompts for professional scientific responses."""
    
    @staticmethod
    def biomedical_analysis_prompt(query: str, context: str, entities: dict) -> str:
        """Generate scientific analysis prompt."""
        
        entity_counts = {k: len(v) for k, v in entities.items() if v}
        
        prompt = f"""You are a senior biomedical researcher with expertise in drug discovery and clinical pharmacology. 

RESEARCH QUERY: {query}

AVAILABLE EVIDENCE:
{context}

ANALYSIS FRAMEWORK:
1. **Mechanistic Analysis**: Examine molecular mechanisms and pathways
2. **Clinical Evidence**: Evaluate efficacy, safety, and therapeutic outcomes  
3. **Comparative Assessment**: Compare interventions based on evidence strength
4. **Limitations**: Acknowledge gaps in available data

RESPONSE FORMAT:
- **Executive Summary**: 2-3 sentences with direct, actionable conclusion
- **Detailed Analysis**: Comprehensive scientific reasoning with:
  • Molecular mechanisms and target pathways
  • Clinical evidence and study outcomes
  • Comparative effectiveness (if applicable)
  • Safety profiles and contraindications
  • Current research gaps and future directions

SCIENTIFIC STANDARDS:
- Use precise medical terminology
- Cite specific entities from the knowledge graph
- Distinguish between established facts and emerging evidence
- Provide confidence levels where appropriate
- Maintain objectivity and acknowledge uncertainties

Please provide a rigorous scientific analysis based on the available graph data."""

        return prompt
    
    @staticmethod
    def comparative_drug_prompt(query: str, context: str, entities: dict) -> str:
        """Specialized prompt for drug comparison queries."""
        
        drugs = entities.get('drugs', [])
        diseases = entities.get('diseases', [])
        
        prompt = f"""You are conducting a systematic review for: {query}

CLINICAL CONTEXT:
- Target Drugs: {[d['name'] for d in drugs[:3]]}
- Therapeutic Areas: {[d['name'] for d in diseases[:3]]}

EVIDENCE BASE:
{context}

SYSTEMATIC REVIEW FRAMEWORK:
1. **Mechanism of Action**: Compare molecular targets and pathways
2. **Clinical Efficacy**: Evaluate therapeutic outcomes and response rates
3. **Safety Profile**: Assess adverse events and contraindications
4. **Patient Selection**: Identify optimal candidate populations
5. **Clinical Positioning**: Recommend treatment sequencing

RESPONSE REQUIREMENTS:
- **Executive Summary**: Clear recommendation with supporting rationale
- **Detailed Analysis**: Evidence-based comparison with specific metrics
- **Clinical Implications**: Practical guidance for therapeutic decision-making

Provide a comprehensive clinical comparison based on the available evidence."""

        return prompt
