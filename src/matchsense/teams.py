"""Team identity: canonical names, user aliases, and provider spellings.

Canonical names are exactly the ones used by ``config/team_aliases.json`` and
the fitted ``SeasonState``, so a resolved team can be fed to the baseline
model without further mapping. User-facing aliases ("Man City", "Spurs") are
listed explicitly; nothing is fuzzy-matched into a club identity.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from src.ingestion.normalization.aliases import load_alias_maps
from src.matchsense.config import PROJECT_ROOT


@dataclass(frozen=True)
class TeamInfo:
    name: str
    aliases: tuple[str, ...]
    fpl_name: str | None = None
    bbc_slug: str | None = None
    football_data_names: tuple[str, ...] = ()
    api_football_names: tuple[str, ...] = ()


# User-facing aliases. "City" alone is deliberately ambiguous and handled by
# ``resolve`` with an explicit note rather than hidden here.
TEAMS: tuple[TeamInfo, ...] = (
    TeamInfo("Arsenal", ("arsenal", "the gunners", "gunners", "afc", "ars"), "Arsenal", "arsenal",
             ("Arsenal FC",), ("Arsenal",)),
    TeamInfo("Aston Villa", ("aston villa", "villa", "avfc", "avl"), "Aston Villa", "aston-villa",
             ("Aston Villa FC",), ("Aston Villa",)),
    TeamInfo("Bournemouth", ("bournemouth", "afc bournemouth", "the cherries", "cherries", "bou"),
             "Bournemouth", "bournemouth", ("AFC Bournemouth",), ("Bournemouth",)),
    TeamInfo("Brentford", ("brentford", "the bees", "bre"), "Brentford", "brentford",
             ("Brentford FC",), ("Brentford",)),
    TeamInfo("Brighton & Hove Albion", ("brighton", "brighton & hove albion", "brighton and hove albion",
                                        "brighton hove albion", "the seagulls", "seagulls", "bha", "bri"),
             "Brighton", "brighton-and-hove-albion", ("Brighton & Hove Albion FC",), ("Brighton",)),
    TeamInfo("Burnley", ("burnley", "the clarets", "bur"), "Burnley", "burnley", ("Burnley FC",), ("Burnley",)),
    TeamInfo("Chelsea", ("chelsea", "the blues", "cfc", "che"), "Chelsea", "chelsea", ("Chelsea FC",), ("Chelsea",)),
    TeamInfo("Coventry City", ("coventry", "coventry city", "sky blues", "cov"), "Coventry City", "coventry-city",
             ("Coventry City FC",), ("Coventry",)),
    TeamInfo("Crystal Palace", ("crystal palace", "palace", "the eagles", "cpfc", "cry"), "Crystal Palace",
             "crystal-palace", ("Crystal Palace FC",), ("Crystal Palace",)),
    TeamInfo("Everton", ("everton", "the toffees", "toffees", "efc", "eve"), "Everton", "everton",
             ("Everton FC",), ("Everton",)),
    TeamInfo("Fulham", ("fulham", "the cottagers", "ful"), "Fulham", "fulham", ("Fulham FC",), ("Fulham",)),
    TeamInfo("Hull City", ("hull", "hull city", "the tigers", "hul"), "Hull City", "hull-city",
             ("Hull City AFC",), ("Hull City",)),
    TeamInfo("Ipswich Town", ("ipswich", "ipswich town", "the tractor boys", "ips"), "Ipswich Town",
             "ipswich-town", ("Ipswich Town FC",), ("Ipswich",)),
    TeamInfo("Leeds United", ("leeds", "leeds united", "lufc", "lee"), "Leeds", "leeds-united",
             ("Leeds United FC",), ("Leeds",)),
    TeamInfo("Leicester City", ("leicester", "leicester city", "the foxes", "lei"), "Leicester",
             "leicester-city", ("Leicester City FC",), ("Leicester",)),
    TeamInfo("Liverpool", ("liverpool", "the reds", "lfc", "liv", "pool"), "Liverpool", "liverpool",
             ("Liverpool FC",), ("Liverpool",)),
    TeamInfo("Manchester City", ("manchester city", "man city", "man. city", "mcfc", "mci", "mancity",
                                 "the citizens", "citizens", "cityzens", "sky blues manchester"),
             "Man City", "manchester-city", ("Manchester City FC",), ("Manchester City",)),
    TeamInfo("Manchester United", ("manchester united", "man united", "man utd", "man u", "mufc", "mun",
                                   "the red devils", "red devils", "united"),
             "Man Utd", "manchester-united", ("Manchester United FC",), ("Manchester United",)),
    TeamInfo("Newcastle United", ("newcastle", "newcastle united", "the magpies", "magpies", "nufc", "new",
                                  "toon"), "Newcastle", "newcastle-united", ("Newcastle United FC",), ("Newcastle",)),
    TeamInfo("Nottingham Forest", ("nottingham forest", "forest", "nottm forest", "nott'm forest", "nffc", "nfo"),
             "Nott'm Forest", "nottingham-forest", ("Nottingham Forest FC",), ("Nottingham Forest",)),
    TeamInfo("Southampton", ("southampton", "saints", "sou"), "Southampton", "southampton",
             ("Southampton FC",), ("Southampton",)),
    TeamInfo("Sunderland", ("sunderland", "the black cats", "black cats", "safc", "sun"), "Sunderland",
             "sunderland", ("Sunderland AFC",), ("Sunderland",)),
    TeamInfo("Tottenham Hotspur", ("tottenham", "tottenham hotspur", "spurs", "thfc", "tot"), "Spurs",
             "tottenham-hotspur", ("Tottenham Hotspur FC",), ("Tottenham",)),
    TeamInfo("West Ham United", ("west ham", "west ham united", "the hammers", "hammers", "whu"), "West Ham",
             "west-ham-united", ("West Ham United FC",), ("West Ham",)),
    TeamInfo("Wolverhampton Wanderers", ("wolves", "wolverhampton", "wolverhampton wanderers", "wol"), "Wolves",
             "wolverhampton-wanderers", ("Wolverhampton Wanderers FC",), ("Wolves",)),
)

# Clubs whose short form is "City"; used only to explain the ambiguity.
CITY_CLUBS = ("Manchester City", "Coventry City", "Hull City", "Leicester City")


def fold(text: str) -> str:
    """Lowercase, strip accents and punctuation (except &), squash spaces."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9&' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class TeamMention:
    team: str
    alias: str
    start: int
    end: int
    ambiguous_with: tuple[str, ...] = ()


@dataclass
class TeamRegistry:
    teams: tuple[TeamInfo, ...] = TEAMS
    _by_alias: dict[str, str] = field(default_factory=dict)
    _by_provider: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        historical = load_alias_maps(PROJECT_ROOT / "config" / "team_aliases.json")
        for info in self.teams:
            for alias in (info.name, *info.aliases):
                self._by_alias[fold(alias)] = info.name
            for spelling in (info.name, info.fpl_name, *info.football_data_names, *info.api_football_names):
                if spelling:
                    self._by_provider[fold(spelling)] = info.name
        # Every historical source spelling in the reviewed alias file.
        for key, canonical in historical.teams.items():
            self._by_provider.setdefault(fold(key), canonical)
            self._by_provider.setdefault(fold(canonical), canonical)

    def get(self, name: str) -> TeamInfo | None:
        return next((t for t in self.teams if t.name == name), None)

    def canonical(self, provider_name: str) -> str | None:
        """Map a provider/source spelling to the canonical model name."""
        key = fold(provider_name)
        if key in self._by_provider:
            return self._by_provider[key]
        if key in self._by_alias:
            return self._by_alias[key]
        stripped = re.sub(r"\b(fc|afc)\b", "", key).strip()
        return self._by_provider.get(stripped) or self._by_alias.get(stripped)

    def find_mentions(self, text: str) -> list[TeamMention]:
        """Longest-alias-first scan; ``City`` alone is flagged as ambiguous."""
        folded = fold(text)
        taken = [False] * len(folded)
        mentions: list[TeamMention] = []
        for alias in sorted(self._by_alias, key=len, reverse=True):
            if len(alias) <= 3 and alias not in ("afc",):
                # Three-letter codes only when typed in capitals in the raw text.
                if not re.search(rf"\b{re.escape(alias.upper())}\b", text):
                    continue
            if alias in ("united", "afc", "new", "sun", "pool"):
                # Too generic on their own; only accept as whole query words
                # when nothing more specific was found.
                continue
            for match in re.finditer(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", folded):
                if any(taken[match.start():match.end()]):
                    continue
                for i in range(match.start(), match.end()):
                    taken[i] = True
                mentions.append(TeamMention(self._by_alias[alias], alias, match.start(), match.end()))
        for match in re.finditer(r"(?<![a-z0-9])city(?:'s)?(?![a-z0-9])", folded):
            if not any(taken[match.start():match.end()]):
                mentions.append(TeamMention("Manchester City", "city", match.start(), match.end(),
                                            ambiguous_with=CITY_CLUBS[1:]))
        if not mentions:
            for match in re.finditer(r"(?<![a-z0-9])united(?:'s)?(?![a-z0-9])", folded):
                mentions.append(TeamMention("Manchester United", "united", match.start(), match.end(),
                                            ambiguous_with=("Newcastle United", "West Ham United", "Leeds United")))
        return sorted(mentions, key=lambda m: m.start)


REGISTRY = TeamRegistry()
