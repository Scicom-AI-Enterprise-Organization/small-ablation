INDUSTRIES = ["telco", "ecommerce", "banking", "insurance", "government", "food_delivery", "healthcare", "utilities"]

ATTACK_TYPES = [
    "jailbreak", "prompt_injection", "bias_exploitation", "social_engineering",
    "policy_bypass", "refund_manipulation", "identity_fraud", "escalation_fishing",
    "misinformation_extraction", "regulatory_threat", "roleplay_manipulation", "emotional_manipulation"
]

LANGUAGES = ["english", "malay", "chinese", "tamil", "mixed"]

SEVERITIES = ["low", "medium", "high", "critical"]

TONES = ["aggressive", "passive_aggressive", "overly_polite", "emotional", "threatening", "casual"]

REQUIRED_FIELDS = [
    "industry", "attack_type", "language", "severity", "tone",
    "system_prompt", "input", "attack_description", "target_behavior",
    "expected_safe_response"
]

MULTITURN_ATTACK_STRATEGIES = [
    "gradual_escalation",       # each turn pushes slightly further until policy breaks
    "trust_building",           # rapport built over early turns, exploited at a key turn
    "context_poisoning",        # false context planted early, weaponised in a later turn
    "persona_hijacking",        # bot convinced to adopt an unrestricted persona across turns
    "persistent_pressure",      # same harmful request repeated with variation until bot capitulates
    "authority_building",       # false authority claim built incrementally across turns
    "distraction_injection",    # normal queries used to distract, then attack injected mid-flow
    "incremental_commitment",   # small commitments extracted early, escalated to larger violation
    "memory_exploitation",      # earlier turns referenced as implicit consent in attack turn
    "social_proof_building",    # prior assistant responses cited as precedent for the attack
]

MULTITURN_REQUIRED_FIELDS = [
    "industry", "attack_type", "attack_strategy", "language", "severity", "tone",
    "system_prompt", "conversation", "attack_turn_index", "turn_count",
    "attack_description", "target_behavior", "expected_safe_response", "failure_mode",
    "evaluation_tags"
]
