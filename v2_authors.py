from __future__ import annotations

# User-supplied Giants-news editorial priors.
# preference is deliberately descriptive rather than numeric; scope controls
# whether a non-beat writer needs a clearly Giants-specific article.
AUTHOR_PRIORS = {
    "andrew baggarly": {"preference": "elite", "scope": "any_giants_relevant"},
    "grant brisbee": {"preference": "fine", "scope": "any_giants_relevant"},
    "alex pavlovic": {"preference": "elite", "scope": "any_giants_relevant"},
    "susan slusser": {"preference": "very_good", "scope": "any_giants_relevant"},
    "john shea": {"preference": "good", "scope": "any_giants_relevant"},
    "maria guardado": {"preference": "good", "scope": "any_giants_relevant"},
    "shayna rubin": {"preference": "elite", "scope": "any_giants_relevant"},
    "justice delos santos": {"preference": "very_good", "scope": "any_giants_relevant"},
    "evan webeck": {"preference": "fine", "scope": "any_giants_relevant"},
    "alex simon": {"preference": "good", "scope": "any_giants_relevant"},
    "kerry crowley": {"preference": "good", "scope": "giants_specific"},
    "tim kawakami": {"preference": "good", "scope": "giants_specific"},

    # National reporters: high-value when the article itself is Giants-specific.
    "jeff passan": {"preference": "national", "scope": "giants_specific"},
    "buster olney": {"preference": "national", "scope": "giants_specific"},
    "jon heyman": {"preference": "national", "scope": "giants_specific"},
    "bob nightengale": {"preference": "national", "scope": "giants_specific"},
    "jon morosi": {"preference": "national", "scope": "giants_specific"},
    "robert murray": {"preference": "national", "scope": "giants_specific"},
    "ken rosenthal": {"preference": "national", "scope": "giants_specific"},
    "evan drellich": {"preference": "national", "scope": "giants_specific"},
}

# Publication-level priors are intentionally light-touch. They are mainly for
# choosing between multiple versions of the same story, not for suppressing a
# legitimate exclusive or useful article on their own.
SOURCE_PRIORS = {
    "sfgate": {
        "preference": "secondary",
        "reason": "mixed editorial signal; some click-driven packaging",
    },
}

AUTHOR_ALIASES = {
    "baggarly": "andrew baggarly",
    "brisbee": "grant brisbee",
    "pavlovic": "alex pavlovic",
    "slusser": "susan slusser",
    "shea": "john shea",
    "guardado": "maria guardado",
    "rubin": "shayna rubin",
    "de los santos": "justice delos santos",
    "delos santos": "justice delos santos",
    "webeck": "evan webeck",
    "simon": "alex simon",
    "crowley": "kerry crowley",
    "kawakami": "tim kawakami",
    "passan": "jeff passan",
    "olney": "buster olney",
    "heyman": "jon heyman",
    "nightengale": "bob nightengale",
    "morosi": "jon morosi",
    "murray": "robert murray",
    "rosenthal": "ken rosenthal",
    "drellich": "evan drellich",
}

SOURCE_ALIASES = {
    "sfgate giants": "sfgate",
}


def normalize_author(name: str) -> str:
    normalized = " ".join((name or "").strip().lower().split())
    return AUTHOR_ALIASES.get(normalized, normalized)


def author_prior(name: str):
    return AUTHOR_PRIORS.get(normalize_author(name))


def normalize_source(name: str) -> str:
    normalized = " ".join((name or "").strip().lower().split())
    return SOURCE_ALIASES.get(normalized, normalized)


def source_prior(name: str):
    return SOURCE_PRIORS.get(normalize_source(name))
