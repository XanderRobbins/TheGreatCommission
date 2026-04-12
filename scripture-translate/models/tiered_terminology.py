"""Tiered terminology system for Bible translation.

Glossary terms are organized in 3 tiers based on override strength:
- Tier 1: Always override (core theology like God, Jesus, Holy Spirit)
- Tier 2: Context-aware override (theological concepts with exceptions)
- Tier 3: No override (let model decide based on context)
"""

from typing import Dict, Set, Optional
from enum import Enum

from models.terminology import TerminologyDB
from utils.logger import get_logger

logger = get_logger(__name__)


class TermTier(Enum):
    """Tier level for terminology override strength."""
    TIER_1 = 1  # Always override
    TIER_2 = 2  # Context-aware override
    TIER_3 = 3  # No override


class TieredTerminologyDB:
    """Tiered glossary system for theological terms.

    Tier 1 (Always Override): Core entities that have one true canonical name
      - God, Jesus, Holy Spirit, Trinity, Father, Son
      - These are never metaphorical or contextual

    Tier 2 (Context-aware): Theological concepts that may have exceptions
      - Righteousness, grace, covenant, salvation, faith, hope, love
      - Apply override in religious contexts; may have secular uses

    Tier 3 (Model Decides): General terms that need context
      - Good, bad, evil, life, death, truth, light, darkness
      - Let the model decide based on sentence context
    """

    # Fixed core theology (always override)
    TIER_1_TERMS = {
        "god": "Bondye",
        "jesus": "Jezi",
        "christ": "Kris",
        "holy spirit": "Sent-Espri",
        "spirit": "Lespri",
        "father": "Papa",
        "son": "Pitit",
        "trinity": "Trinite",
        "lord": "Seyè",
        "almighty": "Tout-Pwisan",
        "divine": "Divin",
        "godly": "Divin",
        "god's": "Bondye a",
    }

    # Contextual theological concepts (override in religious context)
    TIER_2_TERMS = {
        "salvation": "Salvasyon",
        "savior": "Sovè",
        "save": "Sove",
        "saved": "Sove",
        "redeemer": "Delivre",
        "redemption": "Delivrans",
        "grace": "Gras",
        "mercy": "Mizerikòd",
        "judgment": "Jijman",
        "righteousness": "Drajti",  # Fixed: was "dwa" which is too general
        "righteous": "Drajti",      # Fixed: was "dwa"
        "sin": "Peche",
        "repent": "Repanti",
        "forgive": "Padone",
        "faith": "Lafwa",
        "believe": "Kwe",
        "hope": "Espwa",
        "love": "Lanmou",
        "blessed": "Beni",
        "blessing": "Benediksyon",
        "curse": "Madichon",
        "holy": "Sen",
        "sanctify": "Saktifye",
        "covenant": "Kontra",
        "law": "Lalwa",
        "kingdom": "Wayòm",
        "church": "Legliz",
        "sacrifice": "Sak",
        "offering": "Ofrann",
        "prayer": "Priye",
        "pray": "Priye",
        "worship": "Adore",
        "baptism": "Batèm",
        "baptize": "Batize",
        "communion": "Komyon",
        "eucharist": "Ewkarist",
    }

    # General terms (no override)
    TIER_3_TERMS = {
        "good": "Bon",
        "evil": "Mal",
        "wicked": "Mechant",
        "life": "Lavi",
        "death": "Lamò",
        "truth": "Verite",
        "light": "Limyè",
        "darkness": "fènwa",   # Fixed: "Fono" was wrong; standard HC is fènwa/nwa
        "water": "Dlo",
        "bread": "Pan",
        "eternal": "Eternal",  # Fixed: was "etwèl" (means star-like)
    }

    # Community/people terms (mostly no override, contextual)
    COMMUNITY_TERMS = {
        "disciple": "Disip",
        "apostle": "Apot",
        "prophet": "Pwofèt",
        "priest": "Prèt",
        "king": "Wa",
        "israel": "Izrayèl",
        "jew": "Jip",
        "gentile": "Pèp",
        "congregation": "Kongreagasyon",
        "body": "Kò",
    }

    # Afterlife terms (context-aware)
    AFTERLIFE_TERMS = {
        "heaven": "Syèl",
        "hell": "Lanfè",  # Fixed: was "lanfe" (spelling correction)
        "resurrection": "Rezireksyon",  # Fixed: "Rezirekireksyon" was garbled
        "judgment day": "Jou Jijman an",
    }

    def __init__(self, terminology_db: TerminologyDB):
        """Initialize tiered terminology system.

        Args:
            terminology_db: Underlying TerminologyDB to populate with tiered terms.
        """
        self.db = terminology_db
        self.tier_map: Dict[str, TermTier] = {}

        # Populate tier map and terminology DB
        self._initialize_tiers()

    def _initialize_tiers(self) -> None:
        """Populate terminology DB with tiered terms."""
        # Tier 1 (always override)
        for english_term, creole_term in self.TIER_1_TERMS.items():
            self.db.add_term(english_term, "hat_Latn", creole_term, confidence=0.99, override=True)
            self.tier_map[english_term.lower()] = TermTier.TIER_1

        # Tier 2 (context-aware)
        for english_term, creole_term in self.TIER_2_TERMS.items():
            self.db.add_term(english_term, "hat_Latn", creole_term, confidence=0.95, override=True)
            self.tier_map[english_term.lower()] = TermTier.TIER_2

        for english_term, creole_term in self.AFTERLIFE_TERMS.items():
            self.db.add_term(english_term, "hat_Latn", creole_term, confidence=0.93, override=True)
            self.tier_map[english_term.lower()] = TermTier.TIER_2

        for english_term, creole_term in self.COMMUNITY_TERMS.items():
            self.db.add_term(english_term, "hat_Latn", creole_term, confidence=0.90, override=True)
            self.tier_map[english_term.lower()] = TermTier.TIER_2

        # Tier 3 (no override)
        for english_term, creole_term in self.TIER_3_TERMS.items():
            self.db.add_term(english_term, "hat_Latn", creole_term, confidence=0.70, override=True)
            self.tier_map[english_term.lower()] = TermTier.TIER_3

        logger.info(
            f"Initialized tiered terminology: "
            f"Tier1={len(self.TIER_1_TERMS)}, "
            f"Tier2={len(self.TIER_2_TERMS)+len(self.AFTERLIFE_TERMS)+len(self.COMMUNITY_TERMS)}, "
            f"Tier3={len(self.TIER_3_TERMS)}"
        )

    def get_tier(self, english_term: str) -> Optional[TermTier]:
        """Get the tier level for a term.

        Args:
            english_term: The English term to check.

        Returns:
            TermTier enum value, or None if term is not tiered.
        """
        return self.tier_map.get(english_term.lower())

    def get_terms_by_tier(self, tier: TermTier) -> Dict[str, str]:
        """Get all terms in a specific tier.

        Args:
            tier: The tier to retrieve.

        Returns:
            Dictionary mapping English terms to Haitian Creole translations.
        """
        result = {}

        if tier == TermTier.TIER_1:
            result.update(self.TIER_1_TERMS)
        elif tier == TermTier.TIER_2:
            result.update(self.TIER_2_TERMS)
            result.update(self.AFTERLIFE_TERMS)
            result.update(self.COMMUNITY_TERMS)
        elif tier == TermTier.TIER_3:
            result.update(self.TIER_3_TERMS)

        return result

    def should_override(self, english_term: str) -> bool:
        """Check if a term should be forcefully overridden.

        Tier 1 terms are always overridden.
        Tier 2 and 3 terms are not forcefully overridden in this system
        (that's handled by prompt conditioning in the next step).

        Args:
            english_term: The English term to check.

        Returns:
            True if term should be forcefully overridden, False otherwise.
        """
        tier = self.get_tier(english_term)
        return tier == TermTier.TIER_1

    def print_tiers(self) -> None:
        """Log formatted tier information."""
        logger.info("=== Tiered Terminology System ===")
        logger.info(f"Tier 1 (Always Override): {len(self.TIER_1_TERMS)} terms")
        logger.info(f"Tier 2 (Context-aware): {len(self.TIER_2_TERMS) + len(self.AFTERLIFE_TERMS) + len(self.COMMUNITY_TERMS)} terms")
        logger.info(f"Tier 3 (No Override): {len(self.TIER_3_TERMS)} terms")


if __name__ == "__main__":
    from utils.logger import configure_logging

    configure_logging()

    # Test the tiered system
    db = TerminologyDB()
    tiered = TieredTerminologyDB(db)

    tiered.print_tiers()

    # Show some term lookups
    logger.info("\n=== Example Term Lookups ===")
    for term in ["god", "grace", "good", "righteous", "eternal", "hell"]:
        tier = tiered.get_tier(term)
        override = tiered.should_override(term)
        translation = db.lookup(term, "hat_Latn")
        logger.info(f"{term:15s} → {translation:15s} (Tier {tier.value if tier else '?'}, Override: {override})")
