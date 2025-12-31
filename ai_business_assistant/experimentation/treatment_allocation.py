"""
Treatment allocation for A/B testing using consistent hashing.
"""

import hashlib

def get_treatment(user_id: str, experiment_id: str, traffic_split: float = 0.5) -> str:
    """
    Consistently assign a user to a treatment or control group.
    """
    key = f"{experiment_id}:{user_id}"
    hash_val = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    
    # Normalized value between 0 and 1
    normalized_val = (hash_val % 10000) / 10000.0
    
    if normalized_val < traffic_split:
        return "treatment"
    else:
        return "control"
