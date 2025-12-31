"""
Statistical analysis for A/B testing.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any

def calculate_significance(control_conversions: int, control_total: int, 
                          treatment_conversions: int, treatment_total: int) -> Dict[str, Any]:
    """
    Calculate statistical significance, lift, and confidence intervals.
    """
    p_control = control_conversions / control_total if control_total > 0 else 0
    p_treatment = treatment_conversions / treatment_total if treatment_total > 0 else 0
    
    lift = (p_treatment - p_control) / p_control if p_control > 0 else 0
    
    # Z-test for proportions
    p_pooled = (control_conversions + treatment_conversions) / (control_total + treatment_total)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_total + 1/treatment_total))
    
    z_score = (p_treatment - p_control) / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    return {
        "control_rate": p_control,
        "treatment_rate": p_treatment,
        "lift": lift,
        "z_score": z_score,
        "p_value": p_value,
        "significant": p_value < 0.05
    }
