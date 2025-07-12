# CTU Boundary Annotation Guidelines

## Overview
We need to create gold standard annotations for Coherent Thematic Units (CTUs) in welfare scheme documents. This will be used to evaluate our automatic segmentation system.

## What is a CTU?
A Coherent Thematic Unit (CTU) is a contiguous sequence of sentences that discuss the same topic or theme. CTU boundaries occur when there is a clear thematic shift in the discourse.

## Annotation Task
For each document, you need to:
1. Read through the entire document
2. Identify where thematic boundaries occur
3. Mark the sentence index where each boundary occurs

## Boundary Types to Look For

### 1. Topic Shifts
- **Target Population**: Who is eligible for the scheme?
- **Eligibility Criteria**: What are the requirements?
- **Benefits**: What does the scheme provide?
- **Application Procedure**: How to apply?
- **Timeline**: When are deadlines?
- **Contact Information**: Who to contact?

### 2. Discourse Structure Changes
- **Introduction → Details**: General overview to specific information
- **Problem → Solution**: Issues to remedies
- **General → Specific**: Broad categories to specific examples
- **Process Steps**: Sequential procedures

### 3. Entity Changes
- **Different stakeholders**: Government → Beneficiaries → Officials
- **Different locations**: State level → District level → Village level
- **Different time periods**: Current → Historical → Future

## Annotation Rules

### DO Mark Boundaries When:
- The topic clearly changes (e.g., from eligibility to benefits)
- The discourse structure shifts (e.g., from introduction to detailed procedure)
- Different entities become the focus (e.g., from government to beneficiaries)
- The temporal context changes (e.g., from current to historical)

### DON'T Mark Boundaries When:
- The same topic continues with more details
- There are minor elaborations or examples
- The discourse flow is smooth and continuous
- Only the sentence structure changes but topic remains same

## Annotation Format

### Input Format
```json
{
  "doc_id": "scheme_001",
  "sentences": [
    "The Pradhan Mantri Kisan Samman Nidhi scheme provides financial support to farmers.",
    "Under this scheme, eligible farmers receive Rs. 6000 per year.",
    "The amount is transferred directly to their bank accounts in three equal installments.",
    "To be eligible, farmers must own agricultural land.",
    "The land should be in their name or in the name of their family members.",
    "Small and marginal farmers are the primary beneficiaries of this scheme.",
    "The scheme aims to supplement the financial needs of farmers for procuring inputs.",
    "Farmers can use this money for seeds, fertilizers, and other agricultural inputs.",
    "The application process is simple and can be done online.",
    "Farmers need to provide their Aadhaar number and bank account details."
  ]
}
```

### Output Format
```json
{
  "doc_id": "scheme_001",
  "boundaries": [3, 6, 8],
  "notes": "Boundary at 3: eligibility criteria start. Boundary at 6: benefits details. Boundary at 8: application procedure."
}
```

## Quality Control

### Before Starting:
1. Read the entire document first
2. Understand the overall structure
3. Identify the main themes

### During Annotation:
1. Mark boundaries conservatively - only when you're confident
2. Consider the context before and after each potential boundary
3. Think about whether a reader would naturally pause here

### After Annotation:
1. Review your boundaries
2. Check if the segments make sense thematically
3. Ensure you haven't missed obvious boundaries

## Common Pitfalls

### Over-segmentation:
- Don't mark boundaries for every minor topic shift
- Don't split related information that flows naturally

### Under-segmentation:
- Don't miss clear thematic boundaries
- Don't ignore major topic changes

### Inconsistent boundaries:
- Apply the same criteria throughout the document
- Be consistent with similar types of shifts

## Examples

### Good Boundary (Mark it):
```
Sentence 4: "To be eligible, farmers must own agricultural land."
Sentence 5: "The land should be in their name or in the name of their family members."
Sentence 6: "Small and marginal farmers are the primary beneficiaries of this scheme."
```
**Boundary at 5**: Shifts from eligibility criteria to beneficiary description.

### Bad Boundary (Don't mark):
```
Sentence 1: "The scheme provides financial support to farmers."
Sentence 2: "Under this scheme, eligible farmers receive Rs. 6000 per year."
Sentence 3: "The amount is transferred directly to their bank accounts in three equal installments."
```
**No boundary needed**: All sentences discuss the same benefit (financial support).

## Annotation Interface

We'll use Doccano for annotation:
1. Each document will be loaded as a sequence of sentences
2. Click on sentence numbers to mark boundaries
3. Add notes explaining your reasoning
4. Save and move to next document

## Timeline
- Target: 50 documents
- Estimated time: 12 minutes per document
- Total time: ~10 hours
- Quality check: Review 10% of annotations

## Questions?
If you're unsure about a boundary, mark it and add a note. We'll review unclear cases together. 