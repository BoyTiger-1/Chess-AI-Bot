"""
Subscription tier management for rate limiting.
"""

from enum import Enum
from dataclasses import dataclass

class SubscriptionTier(Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"

@dataclass
class RateLimitPolicy:
    requests_per_minute: int
    burst_allowance: float = 1.5
    daily_quota: int = 1000

TIER_POLICIES = {
    SubscriptionTier.FREE: RateLimitPolicy(requests_per_minute=10, daily_quota=100),
    SubscriptionTier.PRO: RateLimitPolicy(requests_per_minute=100, daily_quota=5000),
    SubscriptionTier.ENTERPRISE: RateLimitPolicy(requests_per_minute=1000, daily_quota=1000000)
}

def get_tier_policy(tier: str) -> RateLimitPolicy:
    try:
        return TIER_POLICIES[SubscriptionTier(tier)]
    except (ValueError, KeyError):
        return TIER_POLICIES[SubscriptionTier.FREE]
