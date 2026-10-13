"""The d1 decision workloads, built deterministically so every engine sees the same bytes.

The five shapes follow the d1-3B model card's speed table: one question, three questions over one
state, a ~3.4k-token state, a 384 px image, and 64 states packed together.
"""
from __future__ import annotations

import base64
import random


TICKET = "I was charged twice this month, please refund one of them."

REFUND = {"type": "noul", "instructions": "Is the customer asking for a refund?"}
TEAM = {
    "type": "choice",
    "instructions": "Which team should handle this?",
    "criteria": {
        "billing": "Charges, refunds, invoices",
        "technical": "App or site faults",
        "fraud": "Suspected unauthorised use",
    },
}
URGENCY = {
    "type": "score",
    "instructions": "How urgent is this?",
    "criteria": ["Can wait", "Today", "Blocking the customer now"],
}
CATS = {
    "type": "choice",
    "instructions": "How many cats are there?",
    "criteria": {"one": "One", "two": "Two", "more": "Three or more"},
}
ESCALATE = {"type": "noul", "instructions": "Should this conversation be escalated to a human supervisor?"}

_SUBJECTS = ["my order", "the invoice", "the mobile app", "my account", "the delivery", "the subscription",
             "the refund", "the login page", "the payment", "the support chat"]
_PROBLEMS = ["has not arrived", "shows the wrong amount", "crashes on start", "was locked without notice",
             "is three days late", "renewed without asking", "was never processed", "rejects my password",
             "failed twice", "keeps disconnecting"]
_TONES = ["Please help.", "This is the second time.", "I need this fixed today.", "Thanks in advance.",
          "Nobody has answered my emails.", "Can someone call me?", "I am very disappointed.",
          "Is there any update?"]


def tickets(n: int, seed: int = 7) -> list[str]:
    """`n` distinct short customer messages."""
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < n:
        t = f"Hi, {rng.choice(_SUBJECTS)} {rng.choice(_PROBLEMS)}. {rng.choice(_TONES)}"
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def long_transcript(target_tokens: int = 3400, count_tokens=None, seed: int = 11) -> str:
    """A synthetic support transcript of about `target_tokens` tokens (exact when `count_tokens` is given)."""
    rng = random.Random(seed)
    lines: list[str] = []
    turn = 0
    est = lambda s: count_tokens(s) if count_tokens else len(s) // 4  # noqa: E731
    while est("\n".join(lines)) < target_tokens:
        turn += 1
        who = "Customer" if turn % 2 else "Agent"
        if who == "Customer":
            text = f"Hi, {rng.choice(_SUBJECTS)} {rng.choice(_PROBLEMS)}. {rng.choice(_TONES)}"
        else:
            text = (f"Thanks for waiting. I checked {rng.choice(_SUBJECTS)} and it "
                    f"{rng.choice(['looks normal on our side', 'shows a pending status', 'needs a manual review', 'was flagged by our system'])}. "
                    f"{rng.choice(['I will open a ticket.', 'Please try again in an hour.', 'I have escalated it.', 'Could you confirm your email address?'])}")
        lines.append(f"[{turn:03d}] {who}: {text}")
    return "\n".join(lines)


def image_data_url() -> str:
    from assets import ensure

    return "data:image/jpeg;base64," + base64.b64encode(ensure().read_bytes()).decode()


def build(count_tokens=None) -> dict[str, dict]:
    """name -> {"requests": [request bodies], "concurrent": bool}"""
    return {
        "one": {"requests": [{"state": TICKET, "questions": {"refund": REFUND}}], "concurrent": False},
        "three": {"requests": [{"state": TICKET, "questions": {"refund": REFUND, "team": TEAM, "urgency": URGENCY}}],
                  "concurrent": False},
        "long": {"requests": [{"state": long_transcript(3400, count_tokens), "questions": {"escalate": ESCALATE}}],
                 "concurrent": False},
        "image": {"requests": [{"state": None, "images": [image_data_url()], "questions": {"cats": CATS}}],
                  "concurrent": False},
        "packed": {"requests": [{"state": t, "questions": {"team": TEAM}} for t in tickets(64)], "concurrent": True},
    }


def parity_set(n: int = 48, seed: int = 3) -> list[dict]:
    """A wider deterministic set for engine-vs-reference parity: varied states, every question type."""
    rng = random.Random(seed)
    qs = [REFUND, TEAM, URGENCY, ESCALATE]
    out = []
    for i, t in enumerate(tickets(n, seed=seed)):
        pick = rng.sample(range(len(qs)), k=rng.randint(1, 3))
        out.append({"state": t, "questions": {f"q{j}": qs[j] for j in pick}})
    out.append({"state": None, "images": [image_data_url()], "questions": {"cats": CATS}})
    return out
