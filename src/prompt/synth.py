# src/prompt/synth.py
import json
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re
import random

class PromptSynthesizer:
    """Synthesize prompts from templates, CTU facts, and cultural cues."""
    
    def __init__(self, templates_file: Optional[Path] = None):
        """
        Initialize prompt synthesizer.
        
        Args:
            templates_file: YAML file with prompt templates
        """
        self.templates = self._load_templates(templates_file)
        self.role_order = [
            "target_pop", "eligibility", "benefits", 
            "procedure", "timeline", "contact", "misc"
        ]
    
    def _load_templates(self, templates_file: Optional[Path]) -> Dict:
        """Load prompt templates from YAML file."""
        if templates_file and templates_file.exists():
            with open(templates_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default templates
            return {
                "target_pop": [
                    "A diverse group of {target_pop} people in traditional attire, looking hopeful and engaged",
                    "Portrait of {target_pop} individuals in rural setting, showing community diversity"
                ],
                "eligibility": [
                    "Document verification process with {eligibility_criteria} clearly visible",
                    "Checklist showing {eligibility_criteria} requirements in simple visual format"
                ],
                "benefits": [
                    "Visual representation of {benefits} being provided to beneficiaries",
                    "Before and after comparison showing impact of {benefits} on daily life"
                ],
                "procedure": [
                    "Step-by-step visual guide for {application_process}",
                    "Simple flowchart showing {application_process} with clear icons"
                ],
                "timeline": [
                    "Calendar or timeline showing {important_dates} with clear markers",
                    "Visual timeline of {important_dates} with seasonal context"
                ],
                "contact": [
                    "Contact information displayed clearly with {contact_details}",
                    "Help desk or support center with {contact_details} prominently shown"
                ],
                "misc": [
                    "General information about {topic} presented in accessible visual format",
                    "Educational illustration about {topic} suitable for low-literacy audience"
                ]
            }
    
    def extract_facts(self, ctu_text: str) -> Dict[str, str]:
        """
        Extract key facts from CTU text.
        
        Args:
            ctu_text: Text of the CTU
            
        Returns:
            Dictionary of extracted facts
        """
        facts = {}
        
        # Extract numbers and amounts
        amounts = re.findall(r'Rs\.?\s*(\d+(?:,\d+)*(?:\.\d+)?)', ctu_text)
        if amounts:
            facts['amount'] = f"Rs. {amounts[0]}"
        
        # Extract percentages
        percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', ctu_text)
        if percentages:
            facts['percentage'] = f"{percentages[0]}%"
        
        # Extract dates/years
        years = re.findall(r'(\d{4})', ctu_text)
        if years:
            facts['year'] = years[0]
        
        # Extract target population
        target_patterns = [
            r'farmers?',
            r'women',
            r'students?',
            r'elderly',
            r'disabled',
            r'tribal',
            r'minority'
        ]
        for pattern in target_patterns:
            if re.search(pattern, ctu_text, re.IGNORECASE):
                facts['target_pop'] = re.search(pattern, ctu_text, re.IGNORECASE).group()
                break
        
        # Extract eligibility criteria
        eligibility_keywords = [
            'eligible', 'qualify', 'requirement', 'criteria', 'condition'
        ]
        for keyword in eligibility_keywords:
            if keyword in ctu_text.lower():
                # Extract sentence containing eligibility info
                sentences = ctu_text.split('.')
                for sent in sentences:
                    if keyword in sent.lower():
                        facts['eligibility_criteria'] = sent.strip()
                        break
                break
        
        # Extract benefits
        benefit_keywords = [
            'benefit', 'provide', 'support', 'assistance', 'help'
        ]
        for keyword in benefit_keywords:
            if keyword in ctu_text.lower():
                sentences = ctu_text.split('.')
                for sent in sentences:
                    if keyword in sent.lower():
                        facts['benefits'] = sent.strip()
                        break
                break
        
        return facts
    
    def generate_image_prompt(self, ctu: Dict, cultural_cues: List[Dict] = None) -> str:
        """
        Generate image prompt for a CTU.
        
        Args:
            ctu: CTU dictionary with 'text', 'role', etc.
            cultural_cues: List of cultural snippets (optional)
            
        Returns:
            Generated image prompt
        """
        role = ctu.get('role', 'misc')
        text = ctu.get('text', '')
        
        # Extract facts from CTU
        facts = self.extract_facts(text)
        
        # Get template for this role
        if role in self.templates:
            templates = self.templates[role]
            template = random.choice(templates)
        else:
            template = "Visual representation of {topic} in accessible format"
        
        # Fill template with facts
        prompt = template
        for key, value in facts.items():
            prompt = prompt.replace(f"{{{key}}}", value)
        
        # Add cultural context if available
        if cultural_cues:
            cultural_context = cultural_cues[0]['snippet'] if cultural_cues else ""
            if cultural_context:
                prompt += f", incorporating cultural elements: {cultural_context}"
        
        # Ensure prompt is not too long (max 35 tokens)
        words = prompt.split()
        if len(words) > 35:
            prompt = " ".join(words[:35]) + "..."
        
        return prompt
    
    def generate_caption(self, ctu: Dict, cultural_cues: List[Dict] = None) -> str:
        """
        Generate bilingual caption for a CTU.
        
        Args:
            ctu: CTU dictionary
            cultural_cues: List of cultural snippets (optional)
            
        Returns:
            Generated caption (≤12 words)
        """
        role = ctu.get('role', 'misc')
        text = ctu.get('text', '')
        
        # Extract key information
        facts = self.extract_facts(text)
        
        # Generate English caption based on role
        if role == "target_pop":
            caption = f"Who can apply: {facts.get('target_pop', 'eligible people')}"
        elif role == "eligibility":
            caption = f"Requirements: {facts.get('eligibility_criteria', 'check criteria')}"
        elif role == "benefits":
            caption = f"Benefits: {facts.get('benefits', 'financial support')}"
        elif role == "procedure":
            caption = f"How to apply: {facts.get('application_process', 'simple steps')}"
        elif role == "timeline":
            caption = f"Important dates: {facts.get('important_dates', 'check calendar')}"
        elif role == "contact":
            caption = f"Contact: {facts.get('contact_details', 'get help')}"
        else:
            caption = "Important information about this scheme"
        
        # Limit to 12 words
        words = caption.split()
        if len(words) > 12:
            caption = " ".join(words[:12])
        
        # Add cultural context if available
        if cultural_cues:
            cultural_hint = cultural_cues[0]['snippet'][:50] if cultural_cues else ""
            if cultural_hint:
                caption += f" | {cultural_hint}"
        
        return caption
    
    def synthesize_poster_data(self, ctus: List[Dict], 
                              cultural_retriever=None) -> List[Dict]:
        """
        Synthesize complete poster data for all CTUs.
        
        Args:
            ctus: List of classified CTUs
            cultural_retriever: CulturalRetriever instance (optional)
            
        Returns:
            List of poster data dictionaries
        """
        poster_data = []
        
        # Sort CTUs by canonical role order
        role_to_order = {role: i for i, role in enumerate(self.role_order)}
        sorted_ctus = sorted(ctus, key=lambda x: role_to_order.get(x.get('role', 'misc'), 999))
        
        for i, ctu in enumerate(sorted_ctus):
            # Get cultural cues if retriever available
            cultural_cues = []
            if cultural_retriever:
                lang_counts = ctu.get('lang_counts', {'en': 1})
                cultural_cues = cultural_retriever.query_ctu(ctu['text'], lang_counts)
            
            # Generate prompt and caption
            image_prompt = self.generate_image_prompt(ctu, cultural_cues)
            caption = self.generate_caption(ctu, cultural_cues)
            
            poster_info = {
                'poster_id': i + 1,
                'ctu_id': ctu.get('ctu_id', i + 1),
                'role': ctu.get('role', 'misc'),
                'image_prompt': image_prompt,
                'caption': caption,
                'cultural_cues': cultural_cues,
                'ctu_text': ctu.get('text', ''),
                'lang_counts': ctu.get('lang_counts', {})
            }
            
            poster_data.append(poster_info)
        
        return poster_data
    
    def save_templates(self, output_file: Path) -> None:
        """Save current templates to YAML file."""
        with open(output_file, 'w') as f:
            yaml.dump(self.templates, f, default_flow_style=False, indent=2)
        print(f"Templates saved to {output_file}")

def main():
    """CLI interface for prompt synthesis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Synthesize prompts for CTU-FlowRAG")
    parser.add_argument("--input", required=True, help="Input classified CTU JSON file")
    parser.add_argument("--output", required=True, help="Output poster JSON file")
    parser.add_argument("--templates", help="YAML file with prompt templates")
    parser.add_argument("--cultural-index", help="Path to cultural index directory")
    
    args = parser.parse_args()
    
    # Load CTUs
    with open(args.input, 'r') as f:
        ctus = json.load(f)
    
    # Initialize synthesizer
    synthesizer = PromptSynthesizer(Path(args.templates) if args.templates else None)
    
    # Initialize cultural retriever if available
    cultural_retriever = None
    if args.cultural_index:
        from src.rag.cultural import CulturalRetriever
        cultural_retriever = CulturalRetriever(Path(args.cultural_index))
    
    # Synthesize poster data
    poster_data = synthesizer.synthesize_poster_data(ctus, cultural_retriever)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(poster_data, f, indent=2)
    
    print(f"Poster data saved to {args.output}")
    print(f"Generated {len(poster_data)} posters")

if __name__ == "__main__":
    main() 