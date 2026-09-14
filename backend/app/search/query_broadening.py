import re

_SIZE_TOKENS = {"xxs", "xs", "s", "m", "l", "xl", "xxl", "xxxl", "xxxxl"}
_COLOR_WORDS = {
    "black", "white", "red", "blue", "green", "yellow", "orange", "purple",
    "pink", "grey", "gray", "brown", "beige", "gold", "silver", "navy",
    "maroon", "teal", "cyan", "magenta", "neon", "matte", "glossy", "multicolor",
}


def broadened_queries(title: str) -> list[str]:
    """Derives progressively more generic search queries from a specific product title.

    A reverse-image search can identify a title too specific to have live shopping
    listings (e.g. an exact color/size variant). Stripping size and color tokens, then
    falling back to just the last word or two, usually lands on the general object
    name (e.g. "Axor Apex Hunter Black Neon Blue Helmet-XL" -> ... -> "Helmet"), which
    is far more likely to have live listings.
    """
    without_parens = re.sub(r"[\(\[][^\)\]]*[\)\]]", "", title)
    tokens = [t for t in re.split(r"[\s\-/]+", without_parens.strip()) if t]
    significant = [
        t for t in tokens if t.lower() not in _SIZE_TOKENS and t.lower() not in _COLOR_WORDS
    ]

    queries: list[str] = []

    def add(query: str) -> None:
        query = query.strip()
        if query and query.lower() not in {q.lower() for q in queries}:
            queries.append(query)

    add(" ".join(significant))
    if len(significant) > 2:
        add(" ".join(significant[-2:]))
    if significant:
        add(significant[-1])

    return queries


def most_generic_term(title: str) -> str:
    """Returns the single most generic term derivable from a specific title — its
    plain object category (e.g. "Helmet"), for use as a YOLO training class label.
    """
    queries = broadened_queries(title)
    return queries[-1] if queries else title
