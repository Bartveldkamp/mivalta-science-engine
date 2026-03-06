#!/usr/bin/env python3
"""
Knowledge Card Parser — Structured extraction from GATC Markdown cards.

Parses the 18 canonical knowledge cards into structured Python objects
that the training data generator can use to create grounded examples.

Each card has sections marked by HTML comments:
  <!-- META -->       → concept_id, axis_owner, version, status
  <!-- SCIENCE: * --> → research citations
  <!-- ALG: * -->     → algorithmic tables and rules
  <!-- JOSI: * -->    → athlete-facing explanations
  <!-- CROSS: * -->   → cross-references
  <!-- META: * -->    → boundaries

This parser extracts ALL structured data — tables, rules, thresholds,
Josi language — so the training generator can create examples that are
deeply grounded in the source of truth.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path


KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge" / "gatc"


@dataclass
class Table:
    """A markdown table extracted from a knowledge card."""
    name: str
    headers: list[str]
    rows: list[dict[str, str]]

    def lookup(self, key_col: str, key_val: str) -> dict[str, str] | None:
        """Lookup a row by key column value."""
        for row in self.rows:
            if row.get(key_col, "").strip() == key_val:
                return row
        return None

    def column_values(self, col: str) -> list[str]:
        """Get all values for a column."""
        return [row.get(col, "") for row in self.rows if col in row]


@dataclass
class KnowledgeCard:
    """A parsed GATC knowledge card."""
    concept_id: str
    axis_owner: str
    version: str
    status: str
    filepath: Path

    # Raw markdown
    raw: str

    # Parsed sections
    science_citations: list[dict[str, str]] = field(default_factory=list)
    alg_tables: dict[str, Table] = field(default_factory=dict)
    alg_notes: dict[str, list[str]] = field(default_factory=dict)
    josi_sections: dict[str, str] = field(default_factory=dict)
    cross_references: list[dict[str, str]] = field(default_factory=list)
    boundaries: list[str] = field(default_factory=list)

    def get_table(self, name: str) -> Table | None:
        """Get an ALG table by name."""
        return self.alg_tables.get(name)

    def all_josi_text(self) -> str:
        """Combine all Josi sections into a single text."""
        return "\n\n".join(self.josi_sections.values())

    def all_rules_text(self) -> str:
        """Combine all ALG sections into a summary."""
        parts = []
        for name, table in self.alg_tables.items():
            parts.append(f"### {name}")
            parts.append(f"Headers: {', '.join(table.headers)}")
            parts.append(f"Rows: {len(table.rows)}")
            if name in self.alg_notes:
                for note in self.alg_notes[name]:
                    parts.append(f"  - {note}")
        return "\n".join(parts)


def _parse_table(lines: list[str]) -> Table | None:
    """Parse a markdown table from lines."""
    # Find header row (first row with |)
    header_idx = None
    for i, line in enumerate(lines):
        if "|" in line and not line.strip().startswith("|--"):
            header_idx = i
            break

    if header_idx is None:
        return None

    header_line = lines[header_idx]
    headers = [h.strip() for h in header_line.split("|") if h.strip()]

    if not headers:
        return None

    rows = []
    for line in lines[header_idx + 1:]:
        line = line.strip()
        if not line or not "|" in line:
            break
        if line.replace("|", "").replace("-", "").replace(" ", "") == "":
            continue  # separator row
        cells = [c.strip() for c in line.split("|") if c.strip() or line.count("|") > len(headers)]
        # Handle empty cells
        raw_cells = line.split("|")
        cells = [c.strip() for c in raw_cells[1:-1]] if raw_cells[0].strip() == "" else [c.strip() for c in raw_cells if c.strip()]

        if len(cells) >= len(headers):
            row = {headers[i]: cells[i] for i in range(len(headers))}
            rows.append(row)
        elif cells:
            row = {}
            for i, cell in enumerate(cells):
                if i < len(headers):
                    row[headers[i]] = cell
            if row:
                rows.append(row)

    # Derive table name from context
    return Table(name="", headers=headers, rows=rows)


def _extract_notes(lines: list[str]) -> list[str]:
    """Extract Notes: bullet points from lines."""
    notes = []
    in_notes = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("notes:"):
            in_notes = True
            continue
        if in_notes:
            if stripped.startswith("- "):
                notes.append(stripped[2:])
            elif stripped == "" or stripped.startswith("#") or stripped.startswith("---"):
                break
    return notes


def parse_card(filepath: Path) -> KnowledgeCard:
    """Parse a single knowledge card from markdown file."""
    raw = filepath.read_text()
    lines = raw.split("\n")

    # Extract META header
    concept_id = filepath.stem
    axis_owner = ""
    version = "1.0"
    status = "frozen"

    for line in lines[:15]:
        if line.startswith("concept_id:"):
            concept_id = line.split(":", 1)[1].strip()
        elif line.startswith("axis_owner:"):
            axis_owner = line.split(":", 1)[1].strip()
        elif line.startswith("version:"):
            version = line.split(":", 1)[1].strip()
        elif line.startswith("status:"):
            status = line.split(":", 1)[1].strip()

    card = KnowledgeCard(
        concept_id=concept_id,
        axis_owner=axis_owner,
        version=version,
        status=status,
        filepath=filepath,
        raw=raw,
    )

    # Split into sections by HTML comments
    current_section_type = None
    current_section_name = None
    current_lines: list[str] = []

    def flush_section():
        nonlocal current_lines
        if not current_section_type or not current_lines:
            current_lines = []
            return

        text = "\n".join(current_lines).strip()

        if current_section_type == "SCIENCE":
            # Parse citations table
            table = _parse_table(current_lines)
            if table and table.rows:
                card.science_citations = table.rows

        elif current_section_type == "ALG":
            name = current_section_name or "unnamed"
            # Find section heading
            for line in current_lines:
                if line.startswith("## "):
                    name = line[3:].strip()
                    break
            table = _parse_table(current_lines)
            if table and table.rows:
                table.name = name
                safe_key = re.sub(r'[^a-z0-9_]', '_', name.lower()).strip('_')
                card.alg_tables[safe_key] = table
                notes = _extract_notes(current_lines)
                if notes:
                    card.alg_notes[safe_key] = notes

        elif current_section_type == "JOSI":
            name = current_section_name or "general"
            for line in current_lines:
                if line.startswith("## "):
                    name = line[3:].strip()
                    break
            card.josi_sections[name] = text

        elif current_section_type == "CROSS":
            table = _parse_table(current_lines)
            if table and table.rows:
                card.cross_references = table.rows

        elif current_section_type == "META" and current_section_name == "boundaries":
            for line in current_lines:
                stripped = line.strip()
                if stripped.startswith("- "):
                    card.boundaries.append(stripped[2:])

        current_lines = []

    for line in lines:
        # Check for section markers
        comment_match = re.match(r'<!--\s*(\w+)(?::\s*(\S+))?\s*-->', line.strip())
        if comment_match:
            flush_section()
            current_section_type = comment_match.group(1)
            current_section_name = comment_match.group(2)
        else:
            current_lines.append(line)

    flush_section()

    return card


def load_all_cards() -> dict[str, KnowledgeCard]:
    """Load and parse all 18 GATC knowledge cards."""
    cards = {}
    for md_file in sorted(KNOWLEDGE_DIR.glob("*.md")):
        card = parse_card(md_file)
        cards[card.concept_id] = card
    return cards


# ---------------------------------------------------------------------------
# Convenience accessors for the training generator
# ---------------------------------------------------------------------------

def get_zone_physiology(cards: dict[str, KnowledgeCard]) -> Table | None:
    """Get the zone physiology table."""
    card = cards.get("zone_physiology")
    if card:
        for table in card.alg_tables.values():
            if "zone" in [h.lower() for h in table.headers]:
                return table
    return None


def get_energy_systems(cards: dict[str, KnowledgeCard]) -> Table | None:
    """Get the energy system definitions table."""
    card = cards.get("energy_systems")
    if card:
        for name, table in card.alg_tables.items():
            if "system_id" in table.headers:
                return table
    return None


def get_zone_to_system_map(cards: dict[str, KnowledgeCard]) -> Table | None:
    """Get the zone-to-energy-system mapping table."""
    card = cards.get("energy_systems")
    if card:
        for name, table in card.alg_tables.items():
            if "primary_system" in table.headers and "zone" in table.headers:
                return table
    return None


def get_load_factors(cards: dict[str, KnowledgeCard]) -> dict[str, float]:
    """Get zone load factors as {zone: factor}."""
    zt = get_zone_to_system_map(cards)
    if not zt:
        return {}
    return {row["zone"]: float(row.get("load_factor", 1.0))
            for row in zt.rows if "zone" in row}


def get_readiness_gates(cards: dict[str, KnowledgeCard]) -> Table | None:
    """Get readiness → zone block gates."""
    card = cards.get("load_monitoring")
    if card:
        for name, table in card.alg_tables.items():
            if "readiness_state" in table.headers and "zone_block" in table.headers:
                return table
    return None


def get_goal_archetypes(cards: dict[str, KnowledgeCard]) -> Table | None:
    """Get goal archetype definitions."""
    card = cards.get("goal_demands")
    if card:
        for name, table in card.alg_tables.items():
            if "archetype" in table.headers:
                return table
    return None


def get_meso_dance_slices(cards: dict[str, KnowledgeCard]) -> Table | None:
    """Get meso dance slice model."""
    card = cards.get("meso_dance_policy")
    if card:
        for name, table in card.alg_tables.items():
            if "purpose" in table.headers and "slice_index" in table.headers:
                return table
    return None


def get_feasibility_tiers(cards: dict[str, KnowledgeCard]) -> Table | None:
    """Get feasibility tier thresholds."""
    card = cards.get("feasibility_policy")
    if card:
        for name, table in card.alg_tables.items():
            if "tier" in [h.lower() for h in table.headers]:
                return table
    return None


def get_modifiers(cards: dict[str, KnowledgeCard]) -> dict:
    """Get age and level modifier tables."""
    card = cards.get("modifiers")
    if not card:
        return {}
    return {name: table for name, table in card.alg_tables.items()}


if __name__ == "__main__":
    cards = load_all_cards()
    print(f"Loaded {len(cards)} knowledge cards:\n")
    for cid, card in cards.items():
        tables = list(card.alg_tables.keys())
        josi = list(card.josi_sections.keys())
        print(f"  {cid} ({card.axis_owner})")
        print(f"    ALG tables: {tables}")
        print(f"    JOSI sections: {josi}")
        print(f"    Science citations: {len(card.science_citations)}")
        print()
