# src/role/tag.py
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import openai
from openai import OpenAI
import os

# Role labels for welfare schemes
ROLE_LABELS = [
    "target_pop",      # Target population/beneficiaries
    "eligibility",     # Eligibility criteria
    "benefits",        # Benefits and amounts
    "procedure",       # Application process
    "timeline",        # Important dates/deadlines
    "contact",         # Contact information
    "exclusions",     # Who is NOT eligible / excluded
    "misc"            # Other information
]

class HybridRoleClassifier:
    """Hybrid role classifier with SVM fallback and LLM primary."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize hybrid classifier."""
        self.cache_dir = cache_dir or Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            raise ValueError("OpenAI API key required")
        
        # Initialize SVM pipeline
        self.svm_pipeline = None
        self._train_svm_fallback()
    
    def _train_svm_fallback(self) -> None:
        """Train SVM fallback classifier with synthetic training data."""
        training_data = self._generate_training_data()
        
        texts = [item['text'] for item in training_data]
        labels = [item['label'] for item in training_data]
        
        # Create TF-IDF + SVM pipeline
        self.svm_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
            ('svm', LinearSVC(random_state=42))
        ])
        
        self.svm_pipeline.fit(texts, labels)
    
    def _generate_training_data(self) -> List[Dict[str, str]]:
        """Generate synthetic training data for SVM fallback."""
        training_data = []
        
        # Training examples for each role
        examples = {
            "target_pop": [
                "Farmers who own agricultural land are eligible for this scheme.",
                "Small and marginal farmers are the primary beneficiaries.",
                "The scheme targets individual farmers and their families.",
                "Women entrepreneurs can apply for this scheme.",
                "Students from economically weaker sections are eligible."
            ],
            "eligibility": [
                "To be eligible, applicants must have a valid Aadhaar card.",
                "The land should be in the applicant's name or family members.",
                "Only those with annual income below Rs. 2.5 lakh can apply.",
                "Applicants must be residents of the state for at least 5 years.",
                "The scheme is open to all citizens above 18 years of age."
            ],
            "benefits": [
                "Under this scheme, beneficiaries receive Rs. 6000 per year.",
                "The amount is transferred directly to bank accounts.",
                "Farmers can use this money for seeds and fertilizers.",
                "The scheme provides 50% subsidy on agricultural inputs.",
                "Beneficiaries get free health insurance coverage."
            ],
            "procedure": [
                "The application process is simple and can be done online.",
                "Visit the official portal and fill the application form.",
                "Submit required documents including Aadhaar and bank details.",
                "The application will be processed within 30 days.",
                "Approval is done at the district level."
            ],
            "timeline": [
                "The scheme is operational from April 2023.",
                "Applications are accepted until December 31st.",
                "Benefits are distributed quarterly.",
                "The deadline for submission is March 31st.",
                "Processing takes 15-30 working days."
            ],
            "contact": [
                "For more information, contact the nearest agriculture office.",
                "Call the helpline number 1800-XXX-XXXX for assistance.",
                "Visit the official website for detailed guidelines.",
                "Email support is available at support@scheme.gov.in.",
                "District offices provide in-person assistance."
            ],
            "misc": [
                "The scheme is implemented by the Ministry of Agriculture.",
                "This initiative aims to support rural development.",
                "The program is funded by the central government.",
                "Regular monitoring ensures proper implementation.",
                "The scheme has been extended for another year."
            ],
            "exclusions": [
                "Large commercial farmers are excluded from this subsidy program.",
                "Applicants owning more than 5 hectares of land are not eligible.",
                "Government employees cannot apply under this scheme.",
                "Individuals who have defaulted on bank loans are excluded.",
                "Enterprises engaged in tobacco production are not covered."
            ]
        }
        
        # Add all examples to training data
        for role, texts in examples.items():
            for text in texts:
                training_data.append({"text": text, "label": role})
        
        return training_data
    
    def classify_with_llm(self, text: str) -> Dict[str, Any]:
        """Classify text using LLM."""
        prompt = f"""
You are an expert in analyzing welfare scheme documents. Classify the role of this text:

Available roles:
- target_pop: Information about target population or beneficiaries
- eligibility: Eligibility criteria and requirements
- benefits: Benefits, amounts, and what beneficiaries receive
- procedure: Application process and procedures
- timeline: Important dates, deadlines, and time-related information
- contact: Contact information, helpline numbers, office details
- exclusions: Groups or entities explicitly NOT eligible
- misc: Other information not fitting the above categories

Text: "{text}"

Return only a JSON object:
{{
    "role": "one_of_the_roles_above",
    "confidence": 0.95,
    "reasoning": "brief explanation"
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            
            result = json.loads(content)
            result['method'] = 'llm'
            return result
            
        except Exception as e:
            print(f"LLM classification failed: {e}")
            return None
    
    def classify_with_svm(self, text: str) -> Dict[str, Any]:
        """Classify text using SVM fallback."""
        if self.svm_pipeline is None:
            return {"role": "misc", "confidence": 0.5, "method": "svm_fallback"}
        
        try:
            predicted_role = self.svm_pipeline.predict([text])[0]
            decision_scores = self.svm_pipeline.decision_function([text])[0]
            confidence = min(0.95, max(0.5, abs(decision_scores[ROLE_LABELS.index(predicted_role)]) / 2))
            
            return {
                "role": predicted_role,
                "confidence": confidence,
                "method": "svm",
                "reasoning": f"SVM classification based on TF-IDF features"
            }
        except Exception as e:
            print(f"SVM classification failed: {e}")
            return {"role": "misc", "confidence": 0.5, "method": "svm_error"}
    
    def classify(self, text: str, use_llm: bool = True) -> Dict[str, Any]:
        """Classify text using hybrid approach."""
        if use_llm:
            llm_result = self.classify_with_llm(text)
            if llm_result and llm_result.get('confidence', 0) > 0.7:
                return llm_result
        
        return self.classify_with_svm(text)

def hybrid_classifier(text: str, use_llm: bool = True) -> Dict[str, Any]:
    """Convenience function for hybrid role classification."""
    classifier = HybridRoleClassifier()
    return classifier.classify(text, use_llm)

def tag_ctus(ctu_list: List[Dict[str, Any]], use_llm: bool = True) -> List[Dict[str, Any]]:
    """Tag a list of CTUs with their roles."""
    classifier = HybridRoleClassifier()
    
    tagged_ctus = []
    for ctu in ctu_list:
        classification = classifier.classify(ctu['text'], use_llm)
        
        tagged_ctu = ctu.copy()
        tagged_ctu['role'] = classification['role']
        tagged_ctu['role_confidence'] = classification['confidence']
        tagged_ctu['classification_method'] = classification['method']
        tagged_ctu['classification_reasoning'] = classification.get('reasoning', '')
        
        tagged_ctus.append(tagged_ctu)
    
    return tagged_ctus

if __name__ == "__main__":
    # Test the classifier
    test_text = "Farmers receive Rs. 6000 per year in three installments."
    result = hybrid_classifier(test_text)
    print(f"Text: {test_text}")
    print(f"Role: {result['role']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Method: {result['method']}") 