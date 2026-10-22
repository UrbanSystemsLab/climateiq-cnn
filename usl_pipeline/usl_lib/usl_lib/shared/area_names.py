"""<ISO3>_<STATE>_<City> study area names."""

import re

AREA_NAME_RE = re.compile(r"^[A-Z]{3}_[A-Z]{2}_[A-Za-z0-9]+$")

# Word separators within a city name. Other punctuation is dropped, so
# "St. Louis" becomes "StLouis" and "O'Fallon" becomes "OFallon".
_CITY_WORD_SEPARATORS = re.compile(r"[\s\-‐-―_/]+")
_NON_ALPHANUMERIC = re.compile(r"[^A-Za-z0-9]")


def pascal_case_city(city: str) -> str:
    words = []
    for word in _CITY_WORD_SEPARATORS.split(city.strip()):
        kept = _NON_ALPHANUMERIC.sub("", word)
        if kept:
            words.append(kept[0].upper() + kept[1:])
    if not words:
        raise ValueError(f"City name {city!r} contains no letters or digits.")

    return "".join(words)


def is_canonical_area_name(name: str) -> bool:
    return AREA_NAME_RE.match(name) is not None


def canonical_area_name(country_iso: str, state: str, city: str) -> str:
    country_iso = country_iso.strip().upper()
    state = state.strip().upper()
    name = f"{country_iso}_{state}_{pascal_case_city(city)}"

    if not AREA_NAME_RE.match(name):
        raise ValueError(
            f"Derived study area name {name!r} is not valid. Expected "
            "<ISO3>_<STATE>_<City> with a three letter country code, a two "
            "letter state code and a PascalCase city of letters and digits "
            f"(pattern {AREA_NAME_RE.pattern})."
        )

    return name
