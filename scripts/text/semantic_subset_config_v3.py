"""
Semantic subset construction config (v3).

v3 change vs v1/v2:
  - Expand the "impact" bucket to better match legacy HumanML3D hftext_v0-style selection,
    by including jump/hop/jumping-jacks/jump-rope style cues (high-impact / high-energy).

Everything else stays the same to keep comparisons meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Bucket:
    key: str
    desc_zh: str
    patterns: List[str]


@dataclass(frozen=True)
class Attribute:
    key: str
    desc_zh: str
    patterns: List[str]


# Taxonomy v3 (4 buckets) - with legacy-expanded impact
BUCKETS_V3: List[Bucket] = [
    Bucket(
        key="impact",
        desc_zh="Impact/contact/landing + jumping/bouncing (legacy expansion) (impact/contact/landing/fall/get-up/jump/hop)",
        patterns=[
            # original impact/contact/landing
            r"\bimpact\b",
            r"\b(contact|collide|collision|crash|slam)\b",
            r"\b(hit|hits|hitting)\s+the\s+ground\b",
            r"\b(land|lands|landing)\b",
            r"\b(drop|drops|dropping)\b",
            r"\b(fall|falls|falling)\b",
            r"\b(collapse|collapses|collapsing)\b",
            r"\b(get\s+up|stand\s+up|stands\s+up|standing\s+up)\b",
            # legacy expansion (jump/hop family)
            r"\b(jump|jumps|jumping)\b",
            r"\b(hop|hops|hopping)\b",
            r"\b(leap|leaps|leaping)\b",
            r"\b(bounce|bounces|bouncing)\b",
            r"\b(jump\s+rope|jumping\s+rope)\b",
            r"\b(jumping\s+jacks?|jumpingjack)\b",
            r"\b(jump\s+up\s+and\s+down)\b",
        ],
    ),
    Bucket(
        key="strike",
        desc_zh="Striking/fighting (strike/fight/punch/kick)",
        patterns=[
            r"\b(punch|punches|punching)\b",
            r"\b(kick|kicks|kicking)\b",
            r"\b(hit|hits|hitting)\b",
            r"\b(strike|strikes|striking)\b",
            r"\b(slap|slaps|slapping)\b",
            r"\b(tackle|tackles|tackling)\b",
            r"\b(fight|fights|fighting)\b",
            r"\b(boxing|spar|sparring)\b",
        ],
    ),
    Bucket(
        key="balance",
        desc_zh="Loss of balance/recovery/stumbling (balance/recovery/stumble/slip)",
        patterns=[
            r"\b(balance|unsteady)\b",
            r"\b(lose|loses|losing)\s+balance\b",
            r"\b(regain|regains|regaining)\s+balance\b",
            r"\b(recover|recovers|recovering)\b",
            r"\b(stumble|stumbles|stumbling)\b",
            r"\b(stagger|staggers|staggering)\b",
            r"\b(slip|slips|slipping)\b",
            r"\b(trip|trips|tripping)\b",
            r"\b(wobble|wobbles|wobbling)\b",
            r"\b(sway|sways|swaying)\b",
            r"\b(teeter|teeters|teetering)\b",
            r"\b(tiptoe|tiptoes)\b",
        ],
    ),
    Bucket(
        key="tremor",
        desc_zh="Shaking/tremor/vibration (tremor/shake/jitter)",
        patterns=[
            r"\b(tremor|tremble|trembling)\b",
            r"\b(shake|shakes|shaking|shaky)\b",
            r"\b(shiver|shivers|shivering)\b",
            r"\b(jitter|jitters|jittering)\b",
            r"\b(twitch|twitches|twitching)\b",
            r"\b(quiver|quivers|quivering)\b",
            r"\b(vibrate|vibrates|vibrating|vibration)\b",
            r"\b(convulse|convulses|convulsing)\b",
        ],
    ),
]


# Orthogonal attributes (same as v1)
ATTRIBUTES_V3: List[Attribute] = [
    Attribute(
        key="secondary_rich",
        desc_zh="Concurrent/multiple-action cues (secondary-rich / simultaneous actions)",
        patterns=[
            r"\b(while|whilst)\b",
            r"\b(with)\b",
            r"\b(at\s+the\s+same\s+time)\b",
            r"\b(and\s+then)\b",
            r"\b(before|after)\b",
            r"\b(then)\b",
        ],
    ),
    Attribute(
        key="periodic",
        desc_zh="Periodic/repetitive cues (periodic / cyclic / repetitive)",
        patterns=[
            r"\b(repeat|repeats|repeating|repeatedly)\b",
            r"\b(over\s+and\s+over|again\s+and\s+again)\b",
            r"\b(rhythm|rhythmic)\b",
            r"\b(cycle|cyclic)\b",
            r"\b(back\s+and\s+forth)\b",
            r"\b(walk|walking|run|running|jog|jogging|pace|stepping|step)\b",
        ],
    ),
]


MICRO_MODIFIERS_V3: List[str] = [
    r"\b(subtle|subtly)\b",
    r"\b(slight|slightly)\b",
    r"\b(gentle|gently)\b",
    r"\b(small|little)\b",
    r"\b(barely|faintly)\b",
    r"\b(nervous|nervously)\b",
]


SCORE_WEIGHTS_V3: Dict[str, float] = {
    "bucket_hit": 1.0,
    "micro_modifier": 0.5,
    "secondary_rich": 0.4,
    "periodic": 0.15,
}

