# src/prompt/synth.py
import json
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re
import random

# Thompson-sampling bandit for adaptive template/arm selection
try:
    from src.bandit.bandit_agent import BanditAgent
except ImportError:
    # Fallback stub in case bandit not available
    BanditAgent = None

class PromptSynthesizer:
    """Synthesize prompts from templates, CTU facts, and cultural cues."""
    
    def __init__(self, templates_file: Optional[Path] = None, *, style_prefix: str = "Flat-icon infographic, pastel palette – ", max_prompt_len: int = 250):
        """
        Initialize prompt synthesizer.
        
        Args:
            templates_file: YAML file with prompt templates
        """
        self.templates = self._load_templates(templates_file)
        self.role_order = [
            "target_pop", "eligibility", "benefits", 
            "exclusions", "procedure", "timeline", "contact", "misc"
        ]

        # Prompt style and sanitisation
        self.style_prefix = style_prefix.strip()
        self.max_prompt_len = max_prompt_len

        # Initialise bandit agent (arms = string indices 0-9)
        if BanditAgent:
            self.bandit = BanditAgent([str(i) for i in range(10)])
        else:
            self.bandit = None
    
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
                "exclusions": [
                    "Illustration highlighting who is NOT eligible: {exclusion_criteria}",
                    "Crossed-out icons representing ineligible groups: {exclusion_criteria}"
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

        # Extract exclusion criteria
        exclusion_keywords = ['not eligible', 'excluded', 'cannot apply', 'not entitled']
        for keyword in exclusion_keywords:
            if keyword in ctu_text.lower():
                sentences = ctu_text.split('.')
                for sent in sentences:
                    if keyword in sent.lower():
                        facts['exclusion_criteria'] = sent.strip()
                        break
                break
        
        return facts
    
    def _sanitize_prompt(self, prompt: str) -> str:
        """Clean up prompt: drop unresolved {placeholders}, trim length, strip bad chars."""
        # Remove leftover {placeholder} patterns
        prompt = re.sub(r"\{[^}]+\}", "", prompt)
        # Collapse whitespace
        prompt = re.sub(r"\s+", " ", prompt).strip()
        # Strip markdown/HTML remnants
        prompt = re.sub(r"[`*_>#\[\]{}]", "", prompt)
        # Limit length
        if len(prompt) > self.max_prompt_len:
            prompt = prompt[: self.max_prompt_len].rstrip() + "…"
        return prompt

    def generate_image_prompt(self, ctu: Dict, cultural_cues: List[Dict] = None, *, idx: int = None, total: int = None, prev_role: str = None) -> str:
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
        
        # Get template list for this role
        if role in self.templates:
            templates = self.templates[role]
            # Bandit-driven template index
            if self.bandit:
                arm = self.bandit.choose_arm(role)
                idx = int(arm) % len(templates)
                template = templates[idx]
                # placeholder reward = 0 (to be updated offline)
                self.bandit.update(role, arm, reward=0.0)
            else:
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
        
        # Build bridge/header if we have storyboard context
        if idx is not None and total is not None:
            bridge = f"Poster {idx}/{total} – {role.capitalize()}"
            if prev_role:
                bridge += f" (after {prev_role})"
            prompt = f"{bridge}. {prompt}"

        # Prepend style prefix: always on first poster; on later ones include the saved style token to maintain palette
        if self.style_prefix:
            if idx == 1 or idx is None:
                # Save token for reuse
                self._cached_style = self.style_prefix
                prompt = f"{self.style_prefix} {prompt}"
            else:
                # Reuse previously cached style if not already at front
                if hasattr(self, "_cached_style"):
                    prompt = f"{self._cached_style} {prompt}"

        # Final sanitisation
        prompt = self._sanitize_prompt(prompt)
        
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
        elif role == "exclusions":
            caption = f"Not eligible: {facts.get('exclusion_criteria', 'see details')}"
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
        
        # Sanitize caption and enforce <=80 chars
        caption = self._sanitize_prompt(caption)
        if len(caption) > 80:
            caption = caption[:77].rstrip() + "…"
        return caption
    
    def synthesize_poster_data(self, ctus: List[Dict], cultural_retriever=None, storyboard_mode: bool = False) -> List[Dict]:
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
        if storyboard_mode:
            sorted_ctus = ctus  # assume already ordered via storyboard
        else:
            sorted_ctus = sorted(ctus, key=lambda x: role_to_order.get(x.get('role', 'misc'), 999))
        
        seen = set()
        total = len(sorted_ctus)
        prev_role = None
        for i, ctu in enumerate(sorted_ctus):
            # Get cultural cues if retriever available
            cultural_cues = []
            if cultural_retriever:
                lang_counts = ctu.get('lang_counts', {'en': 1})
                cultural_cues = cultural_retriever.query_ctu(ctu['text'], lang_counts)
            
            # Generate prompt and caption
            image_prompt = self.generate_image_prompt(ctu, cultural_cues, idx=i+1, total=total, prev_role=prev_role)
            caption = self.generate_caption(ctu, cultural_cues)
            
            key = (ctu.get('role', 'misc'), caption)
            if key in seen:
                continue

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
            seen.add(key)
            prev_role = poster_info['role']
        
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