#!/usr/bin/env python3
"""
MiValta Knowledge Selector — picks relevant coaching context cards for a conversation turn.

Used in two places:
1. Training data preparation: inject [KNOWLEDGE] blocks into training examples
2. On-device inference: Kotlin app picks cards and injects before model call

Design: simple keyword + topic matching. No embeddings, no vectors.
The 114 cards are small enough (~153 KB) that brute-force matching is instant.

Usage:
    from knowledge_selector import KnowledgeSelector

    selector = KnowledgeSelector.from_json("knowledge/generated/knowledge.json")
    cards = selector.select(
        user_message="What is Zone 2 and why should I train there?",
        action="answer_question",
        sport="run",
        max_cards=3,
    )
    # Returns list of {"id": ..., "content": ...} dicts
"""

import json
import re
from pathlib import Path
from typing import Optional


class KnowledgeSelector:
    """Select relevant knowledge cards for a conversation turn."""

    def __init__(self, entries: list[dict]):
        self.entries = entries
        self._index = {e["id"]: e for e in entries}

    @classmethod
    def from_json(cls, path: str | Path) -> "KnowledgeSelector":
        """Load from knowledge.json."""
        with open(path) as f:
            data = json.load(f)
        return cls(data["entries"])

    def select(
        self,
        user_message: str,
        action: Optional[str] = None,
        sport: Optional[str] = None,
        max_cards: int = 3,
    ) -> list[dict]:
        """
        Select the most relevant knowledge cards for this turn.

        Args:
            user_message: The athlete's message text
            action: Interpreter action (create_workout, explain, answer_question, etc.)
            sport: Sport context (run, bike, ski, etc.)
            max_cards: Maximum number of cards to return (default 3)

        Returns:
            List of {"id": str, "content": str} dicts, highest relevance first
        """
        msg_lower = user_message.lower()
        msg_words = set(re.findall(r'\b\w+\b', msg_lower))

        scored = []
        for entry in self.entries:
            score = self._score_entry(entry, msg_lower, msg_words, action, sport)
            if score > 0:
                scored.append((score, entry))

        # Sort by score descending, then by specificity (prefer non-overview)
        scored.sort(key=lambda x: (x[0], x[1]["section"] != "overview"), reverse=True)

        results = []
        seen_cards = set()
        for score, entry in scored:
            if len(results) >= max_cards:
                break
            # Avoid duplicate cards from same domain unless highly relevant
            card = entry["card"]
            if card in seen_cards and score < 5:
                continue
            seen_cards.add(card)
            results.append({"id": entry["id"], "content": entry["content"]})

        return results

    def _score_entry(
        self,
        entry: dict,
        msg_lower: str,
        msg_words: set[str],
        action: Optional[str],
        sport: Optional[str],
    ) -> float:
        """Score a single entry for relevance. Higher = more relevant."""
        score = 0.0

        # Sport match: strong signal
        entry_sport = entry.get("sport")
        if sport and entry_sport:
            if sport == entry_sport:
                score += 3.0
            else:
                # Wrong sport: heavy penalty
                score -= 5.0

        # Keyword matching: count how many entry keywords appear in message
        # Use substring matching to handle plurals (interval/intervals) and compounds
        entry_keywords = entry.get("keywords", [])
        keyword_hits = 0
        for kw in entry_keywords:
            if " " in kw:
                # Multi-word keyword: check as phrase
                if kw in msg_lower:
                    keyword_hits += 2  # Phrase matches are stronger
            else:
                if kw in msg_lower:
                    keyword_hits += 1

        score += min(keyword_hits * 1.0, 6.0)  # Cap keyword contribution

        # Section relevance: overview cards are good defaults
        section = entry.get("section", "")
        if section == "overview" and keyword_hits > 0:
            score += 1.0
        if section == "safety" and keyword_hits > 0:
            score += 0.5

        # Action-specific boosts
        if action == "answer_question":
            # Knowledge questions benefit from educational content
            if section in ("overview", "all_zones", "models", "science_of_structure"):
                score += 0.5
        elif action == "create_workout":
            # Workout creation benefits from structure + zone cards
            topics = entry.get("topics", [])
            if "workout_structure" in topics or "intervals" in topics:
                score += 1.0
        elif action == "explain":
            # Explanation benefits from contextual cards
            pass

        # Persona cards are internal — exclude from athlete-facing injection
        if entry["card"] == "josi_personas_v1":
            score = 0.0
        if entry["card"] == "planner_policy_v4":
            score = 0.0

        # Safety content boost when message mentions concerning topics
        safety_words = {"pain", "injury", "sick", "illness", "hurt", "stop",
                        "medical", "chest", "dizzy", "pijn", "ziek"}
        if msg_words & safety_words and section == "safety":
            score += 2.0

        # Beginner/senior boost when context suggests it
        beginner_words = {"beginner", "new", "start", "first", "begin", "nieuw"}
        if msg_words & beginner_words and "beginner" in entry.get("topics", []):
            score += 2.0

        senior_words = {"senior", "elderly", "old", "70", "80", "90"}
        if msg_words & senior_words and "seniors" in entry.get("topics", []):
            score += 2.0

        return score

    def format_knowledge_block(self, cards: list[dict]) -> str:
        """Format selected cards as a [KNOWLEDGE] text block for prompt injection."""
        if not cards:
            return ""

        parts = ["[KNOWLEDGE]"]
        for card in cards:
            parts.append(card["content"])
        return "\n\n".join(parts)

    def get_by_id(self, entry_id: str) -> Optional[dict]:
        """Get a specific entry by its ID."""
        entry = self._index.get(entry_id)
        if entry:
            return {"id": entry["id"], "content": entry["content"]}
        return None
