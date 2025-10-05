"""
Compatibility allowlist for role-edge type combinations.

Implements the frozen allowlist that determines which role pairs can be connected
by which edge types, enforcing the policy constraints.
"""

from typing import Dict, Set, Tuple

# Role to ID mapping (frozen order)
ROLE_TO_ID = {
    "ContextObjective": 0,
    "BenefitsAssistance": 1, 
    "Eligibility": 2,
    "ApplicationProcess": 3,
    "TimelineFrequency": 4,
    "AuthoritiesGovernance": 5,
    "DefinitionsReferences": 6
}

ID_TO_ROLE = {v: k for k, v in ROLE_TO_ID.items()}

# Edge types
EDGE_TYPES = [
    "PRECEDES",
    "SEGMENT_CONTINUATION", 
    "PREREQUISITE_OF",
    "ENABLES",
    "CAP_LIMITS",
    "RATE_SPEC",
    "ADMINISTERED_BY",
    "TIMELINE_FOR"
]

# Frozen allowlist: (role_i, edge_type, role_j) -> allowed
# Based on the policy constraints from the design freeze
ALLOWLIST: Dict[Tuple[str, str, str], bool] = {
    # ContextObjective connections
    ("ContextObjective", "PRECEDES", "ContextObjective"): True,
    ("ContextObjective", "SEGMENT_CONTINUATION", "ContextObjective"): True,
    ("ContextObjective", "PREREQUISITE_OF", "BenefitsAssistance"): True,
    ("ContextObjective", "ENABLES", "BenefitsAssistance"): True,
    ("ContextObjective", "PREREQUISITE_OF", "Eligibility"): True,
    ("ContextObjective", "PREREQUISITE_OF", "ApplicationProcess"): True,
    
    # BenefitsAssistance connections
    ("BenefitsAssistance", "PRECEDES", "BenefitsAssistance"): True,
    ("BenefitsAssistance", "SEGMENT_CONTINUATION", "BenefitsAssistance"): True,
    ("BenefitsAssistance", "PREREQUISITE_OF", "Eligibility"): True,
    ("BenefitsAssistance", "ENABLES", "ApplicationProcess"): True,
    ("BenefitsAssistance", "ADMINISTERED_BY", "AuthoritiesGovernance"): True,
    
    # Eligibility connections
    ("Eligibility", "PRECEDES", "Eligibility"): True,
    ("Eligibility", "SEGMENT_CONTINUATION", "Eligibility"): True,
    ("Eligibility", "PREREQUISITE_OF", "ApplicationProcess"): True,
    ("Eligibility", "ENABLES", "ApplicationProcess"): True,
    ("Eligibility", "CAP_LIMITS", "ApplicationProcess"): True,
    ("Eligibility", "RATE_SPEC", "ApplicationProcess"): True,
    
    # ApplicationProcess connections
    ("ApplicationProcess", "PRECEDES", "ApplicationProcess"): True,
    ("ApplicationProcess", "SEGMENT_CONTINUATION", "ApplicationProcess"): True,
    ("ApplicationProcess", "TIMELINE_FOR", "TimelineFrequency"): True,
    ("ApplicationProcess", "ADMINISTERED_BY", "AuthoritiesGovernance"): True,
    
    # TimelineFrequency connections
    ("TimelineFrequency", "PRECEDES", "TimelineFrequency"): True,
    ("TimelineFrequency", "SEGMENT_CONTINUATION", "TimelineFrequency"): True,
    
    # AuthoritiesGovernance connections
    ("AuthoritiesGovernance", "PRECEDES", "AuthoritiesGovernance"): True,
    ("AuthoritiesGovernance", "SEGMENT_CONTINUATION", "AuthoritiesGovernance"): True,
    
    # DefinitionsReferences connections
    ("DefinitionsReferences", "PRECEDES", "DefinitionsReferences"): True,
    ("DefinitionsReferences", "SEGMENT_CONTINUATION", "DefinitionsReferences"): True,
    ("DefinitionsReferences", "PREREQUISITE_OF", "ContextObjective"): True,
    ("DefinitionsReferences", "ENABLES", "ContextObjective"): True,
}

def is_compatible(role_i: str, edge_type: str, role_j: str) -> bool:
    """
    Check if a role-edge-role combination is allowed by the policy.
    
    Args:
        role_i: Source role name
        edge_type: Edge type name  
        role_j: Target role name
        
    Returns:
        True if the combination is allowed, False otherwise
    """
    return ALLOWLIST.get((role_i, edge_type, role_j), False)

def get_compatible_edges(role_i: str) -> Set[Tuple[str, str]]:
    """
    Get all (edge_type, role_j) pairs that are compatible with role_i.
    
    Args:
        role_i: Source role name
        
    Returns:
        Set of (edge_type, role_j) tuples that are compatible
    """
    compatible = set()
    for (src_role, edge_type, tgt_role), allowed in ALLOWLIST.items():
        if src_role == role_i and allowed:
            compatible.add((edge_type, tgt_role))
    return compatible

def get_role_id(role: str) -> int:
    """Get numeric ID for a role name."""
    return ROLE_TO_ID[role]

def get_role_name(role_id: int) -> str:
    """Get role name for a numeric ID."""
    return ID_TO_ROLE[role_id]

def validate_edge(role_i: str, edge_type: str, role_j: str) -> bool:
    """
    Validate an edge against the allowlist and additional constraints.
    
    Args:
        role_i: Source role
        edge_type: Edge type
        role_j: Target role
        
    Returns:
        True if edge is valid, False otherwise
    """
    # Check allowlist
    if not is_compatible(role_i, edge_type, role_j):
        return False
        
    # Additional constraints
    # PRECEDES can only be within same section (handled elsewhere)
    # Out-degree cap of 2 (handled elsewhere)
    
    return True

