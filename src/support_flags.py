"""Phrase flags for human review, separate from model inference."""

import re

_PHRASES = re.compile(
    r"\bsuicid(?:e|al)\b|\bkill myself\b|\bend it all\b|\bdon't want to live\b|"
    r"\bbetter off dead\b|\bwant to die\b|\bself[\s-]?harm\b|\bcut myself\b|"
    r"\bend my life\b|\bno reason to live\b", re.I
)


def support_mentions(text):
    """Handle explicit denials/history/other-person context without claiming diagnosis."""
    text = text.replace("\u2019", "'").lower()
    flags = []
    for sentence in re.split(r"[.!?\n]+|\bbut\b|\bhowever\b", text):
        for match in _PHRASES.finditer(sentence):
            prefix = sentence[max(0, match.start() - 85):match.start()]
            denied = re.search(
                r"(?:\bnot|\bdon't|\bnever|\bno longer|\bnot currently|\bnot feeling|"
                r"\bdo not feel|\bdon't feel|\bdeny|\bdenies|\bno|"
                r"\bnot having|\bno thoughts of|\bnot thinking about|\bdon't want to)\s*$", prefix
            ) or re.search(r"\b(?:not|never)\s+(?:want(?:ing)? to|going to|planning to)\s*$", prefix)
            if denied:
                continue
            contextual = bool(re.search(
                r"\b(?:used to|previously|in the past|years? ago|last year|history of|"
                r"my friend|my brother|my sister|my parent|my roommate|someone else|"
                r"a friend|research|article|lecture|studying|prevention)\b", sentence
            ))
            flags.append({
                "category": "Support review", "label": "Support-related language",
                "description": "Past, third-person, or educational context; clarify in conversation." if contextual
                               else "Potential current support need; check in directly with the student.",
                "is_critical": not contextual, "context_review": contextual,
                "severity": "Review" if contextual else "Priority",
                "badge_color": "#b45309" if contextual else "#b91c1c", "text_color": "#ffffff",
            })
    return flags[:1] if not any(f["is_critical"] for f in flags) else [next(f for f in flags if f["is_critical"])]
