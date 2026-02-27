#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime
import json
import re
import time
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup


USER_AGENT = "WildForagerBot/0.9 (Wikipedia + Wikibooks Bestimmungsbuch; CC BY-SA attribution)"
HEADERS = {"User-Agent": USER_AGENT}

# --- Wikipedia (DE) ---
WIKI_API = "https://de.wikipedia.org/w/api.php"

# --- Wikibooks (DE) ---
WIKIBOOKS_API = "https://de.wikibooks.org/w/api.php"
WIKIBOOKS_WIKI = "https://de.wikibooks.org/wiki/"
WIKIBOOKS_PREFIX = "Bestimmungsbuch_Pflanzen_Mitteleuropas/"  # underscore form used in URLs


SECTION_MAP = {
    "Beschreibung": "morphology",
    "Merkmale": "morphology",
    "Morphologie": "morphology",
    "Blätter": "leaves",
    "Blüten": "flowers",
    "Früchte": "fruit",
    "Vorkommen": "habitat",
    "Standort": "habitat",
    "Verbreitung": "distribution",
    "Verwechslungsmöglichkeiten": "confusion_species",
    "Ökologie": "ecology",
}


# -----------------------
# Helpers
# -----------------------

def requests_get(session: requests.Session, url: str, *, params=None, timeout=30) -> requests.Response:
    r = session.get(url, params=params, headers=HEADERS, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\[\s*\d+(?:\.\d+)?\s*\]", "", text)  # [1], [1.1]
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("↑", "").strip()


def normalize_signature(text: str, max_len: int = 600) -> str:
    t = (text or "").lower()
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"[^\wäöüß]+", "", t, flags=re.UNICODE)
    return t[:max_len]


def normalize_name_for_match(name: str) -> str:
    """
    Match 'Spitz-Ahorn' ~ 'Spitzahorn' etc.
    Lowercase, replace umlauts, remove whitespace/punctuation/hyphens.
    """
    s = (name or "").lower().strip()
    s = s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    s = re.sub(r"[\s\-_–—]+", "", s)
    s = re.sub(r"[^\w]+", "", s)
    return s


def load_names_json(path: str) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit("names_de.json must be a JSON object mapping scientific_name -> german_name.")

    out: dict[str, str] = {}
    for k, v in data.items():
        if isinstance(k, str) and isinstance(v, str):
            ks = k.strip()
            vs = v.strip()
            if ks:
                out[ks] = vs
    return out


# -----------------------
# Wikipedia (DE): traits (deduped)
# -----------------------

def wiki_resolve_title(session: requests.Session, name: str):
    params = {
        "action": "query",
        "format": "json",
        "redirects": 1,
        "prop": "info",
        "inprop": "url",
        "titles": name,
    }
    data = requests_get(session, WIKI_API, params=params).json()
    page = next(iter(data["query"]["pages"].values()))
    if "missing" in page:
        return None
    return page["title"], page["fullurl"]


def wiki_get_sections(session: requests.Session, title: str):
    params = {"action": "parse", "format": "json", "page": title, "prop": "sections"}
    sections = requests_get(session, WIKI_API, params=params).json().get("parse", {}).get("sections", [])

    numbers = [s.get("number") for s in sections if s.get("number")]
    parent_numbers = set()
    for n in numbers:
        prefix = n + "."
        if any(x.startswith(prefix) for x in numbers):
            parent_numbers.add(n)

    for s in sections:
        s["_is_parent"] = bool(s.get("number") in parent_numbers)

    return sections


def wiki_get_section_paragraphs(session: requests.Session, title: str, index: str) -> list[str]:
    params = {"action": "parse", "format": "json", "page": title, "prop": "text", "section": index}
    parse = requests_get(session, WIKI_API, params=params).json().get("parse", {})
    html = parse.get("text", {}).get("*", "")
    if not html:
        return []

    soup = BeautifulSoup(html, "lxml")
    out = []
    for p in soup.find_all("p"):
        t = clean_text(p.get_text(" ", strip=True))
        if t:
            out.append(t)
    return out


def fetch_wikipedia_traits(session: requests.Session, wikipedia_query: str):
    resolved = wiki_resolve_title(session, wikipedia_query)
    if not resolved:
        return None

    title, url = resolved
    sections = wiki_get_sections(session, title)

    traits: dict[str, str] = {}
    seen_para: set[str] = set()
    seen_trait_sig: dict[str, set[str]] = {}

    for sec in sections:
        heading = (sec.get("line") or "").strip()
        if heading not in SECTION_MAP:
            continue

        if sec.get("_is_parent"):
            continue

        normalized = SECTION_MAP[heading]
        paras = wiki_get_section_paragraphs(session, title, sec.get("index"))
        if not paras:
            continue

        kept = []
        for para in paras:
            sig = normalize_signature(para)
            if not sig or sig in seen_para:
                continue
            seen_para.add(sig)
            kept.append(para)

        if not kept:
            continue

        content = " ".join(kept).strip()
        content_sig = normalize_signature(content, max_len=1200)

        seen_trait_sig.setdefault(normalized, set())
        if content_sig in seen_trait_sig[normalized]:
            continue
        seen_trait_sig[normalized].add(content_sig)

        traits[normalized] = (traits.get(normalized, "") + " " + content).strip() if normalized in traits else content

    return {
        "title": title,
        "url": url,
        "traits": traits,
        "license": {
            "name": "CC BY-SA 4.0",
            "source": "German Wikipedia",
            "requirements": ["attribution", "share-alike"],
        },
        "attribution_minimum": f"Wikipedia-Autor*innen, Artikel „{title}“, CC BY-SA 4.0, {url}",
    }


# -----------------------
# Wikibooks Bestimmungsbuch: search + verify
# -----------------------

def wikibooks_search(session: requests.Session, query: str, limit: int = 15) -> list[str]:
    """
    Return candidate page titles (strings).
    """
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srlimit": limit,
        "srprop": "",
        "srsearch": query,
    }
    data = requests_get(session, WIKIBOOKS_API, params=params).json()
    hits = data.get("query", {}).get("search", [])
    return [h.get("title", "") for h in hits if h.get("title")]


def wikibooks_parse_html(session: requests.Session, title: str) -> str:
    params = {"action": "parse", "format": "json", "page": title, "prop": "text", "disableeditsection": 1}
    data = requests_get(session, WIKIBOOKS_API, params=params).json()
    return data.get("parse", {}).get("text", {}).get("*", "")


def wikibooks_title_to_url(title: str) -> str:
    # MediaWiki titles use spaces; URLs use underscores
    return WIKIBOOKS_WIKI + quote(title.replace(" ", "_"), safe=":/()?=&%#._-")


def wikibooks_extract_species_section(html: str, scientific_name: str, german_name: str, max_chars: int = 3500) -> dict:
    """
    Find a heading like 'Spitzahorn (Acer platanoides)' and extract everything until the next heading of same level.
    Then parse:
      - outgoing links (Wikipedia/Commons/FloraWeb)
      - Hauptgruppen path
      - Merkmale block
    """
    soup = BeautifulSoup(html, "lxml")

    sci_norm = normalize_name_for_match(scientific_name)
    de_norm = normalize_name_for_match(german_name)

    # Find best matching heading
    heading_tags = ["h2", "h3", "h4"]
    chosen = None
    chosen_level = None

    for h in soup.find_all(heading_tags):
        t = clean_text(h.get_text(" ", strip=True))
        if not t:
            continue
        tn = normalize_name_for_match(t)
        if sci_norm and sci_norm in tn:
            chosen = h
            chosen_level = h.name
            break
        if de_norm and de_norm in tn:
            chosen = h
            chosen_level = h.name
            break

    if not chosen:
        return {"error": "species_heading_not_found"}

    # Collect nodes until next heading of same level
    nodes = []
    for sib in chosen.find_all_next():
        if sib.name == chosen_level:
            break
        nodes.append(sib)

    # Extract plain text lines for lightweight parsing
    lines = []
    for n in nodes:
        if n.name in ["p", "li"]:
            t = clean_text(n.get_text(" ", strip=True))
            if t:
                lines.append(t)

    full_text = " ".join(lines).strip()
    if max_chars and len(full_text) > max_chars:
        full_text = full_text[:max_chars].rstrip() + "…"

    # Extract links (Wikipedia/Commons/FloraWeb) from anchors in this section
    links = []
    for n in nodes:
        for a in getattr(n, "find_all", lambda *args, **kwargs: [])("a", href=True):
            label = clean_text(a.get_text(" ", strip=True))
            href = a["href"]
            if href.startswith("/wiki/"):
                href = "https://de.wikibooks.org" + href
            links.append({"label": label, "url": href})

    # Keep only likely “resource links”
    resource_links = []
    for l in links:
        lab = (l["label"] or "").lower()
        if any(k in lab for k in ["wikipedia", "commons", "floraweb", "karte"]):
            resource_links.append(l)

    # Find Hauptgruppen path line
    hauptgruppen_path = None
    for ln in lines:
        if "hauptgruppen" in ln.lower() and "=>" in ln:
            hauptgruppen_path = ln
            break

    # Extract Merkmale block:
    # Find "Merkmale" line, then collect subsequent lines until a new section keyword appears
    merkmale = []
    start = None
    for i, ln in enumerate(lines):
        if ln.strip().lower() == "merkmale":
            start = i + 1
            break
        # Some pages have "Merkmale:" inline
        if ln.lower().startswith("merkmale"):
            start = i
            break

    if start is not None:
        for ln in lines[start:]:
            lnl = ln.lower().strip()
            if lnl in {"lebensraum", "verwendung", "verwechslung", "giftigkeit", "quellen", "links"}:
                break
            # Stop if it looks like another heading-ish marker
            if lnl.endswith(":") and len(ln) < 40:
                break
            merkmale.append(ln)
            if len(merkmale) >= 12:
                break

    return {
        "heading": clean_text(chosen.get_text(" ", strip=True)),
        "resource_links": resource_links,
        "hauptgruppen_path": hauptgruppen_path,
        "merkmale": merkmale,
        "text": full_text,
    }

def fetch_wikibooks_bestimmungsbuch(session: requests.Session, scientific_name: str, german_name: str):
    """
    Robust approach:
    - search by scientific name (best)
    - search by normalized german name without hyphen/punctuation (fallback)
    - filter results to Bestimmungsbuch pages
    - verify by content
    """
    queries = []
    if scientific_name:
        queries.append(scientific_name)
    if german_name:
        # remove punctuation for search friendliness
        gn = re.sub(r"[\-_/]", " ", german_name)
        queries.append(gn)

    candidate_titles = []
    for q in queries:
        candidate_titles.extend(wikibooks_search(session, q, limit=20))

    # filter to book pages only
    candidates = []
    for t in candidate_titles:
        if t.replace(" ", "_").startswith(WIKIBOOKS_PREFIX):
            if t not in candidates:
                candidates.append(t)

    if not candidates:
        return {
            "error": "no_candidates_in_bestimmungsbuch",
            "title": None,
            "url": None,
        }

    # verify candidates by fetching html and checking for names
    sci_norm = normalize_name_for_match(scientific_name)
    de_norm = normalize_name_for_match(german_name)

    verified = []
    for t in candidates[:8]:  # cap requests
        html = wikibooks_parse_html(session, t)
        if not html:
            continue
        page_text = clean_text(BeautifulSoup(html, "lxml").get_text(" ", strip=True))
        pt_norm = normalize_name_for_match(page_text)

        score = 0
        if sci_norm and sci_norm in pt_norm:
            score += 3
        if de_norm and de_norm in pt_norm:
            score += 2

        if score > 0:
            verified.append((score, t, html))

    if not verified:
        # at least return a best-guess page (first candidate)
        best = candidates[0]
        return {
            "error": "candidates_found_but_not_verified",
            "title": best,
            "url": wikibooks_title_to_url(best),
        }

    verified.sort(key=lambda x: (-x[0], x[1]))
    score, best_title, best_html = verified[0]

    extracted = wikibooks_extract_species_section(best_html, scientific_name, german_name)

    return {
        "title": best_title,
        "url": wikibooks_title_to_url(best_title),
        "score": score,
        "extracted": extracted,
        "license": {
            "name": "CC BY-SA",
            "source": "Wikibooks (de)",
            "requirements": ["attribution", "share-alike"],
        },
        "attribution_minimum": f"Wikibooks-Autor*innen, „{best_title}“, CC BY-SA, {wikibooks_title_to_url(best_title)}",
        "method": "search_and_verify",
    }


# -----------------------
# Main
# -----------------------

def process(names_json_path: str, output_json: str, delay: float = 1.0, prefer_german_name_for_wikipedia: bool = True):
    names_map = load_names_json(names_json_path)
    session = requests.Session()
    today = str(datetime.date.today())

    results = []

    for scientific_name, german_name in names_map.items():
        if not scientific_name:
            continue

        wiki_query = german_name if (prefer_german_name_for_wikipedia and german_name) else scientific_name
        print("Processing:", scientific_name, f"(DE: {german_name}; wiki query: {wiki_query})")

        # Wikipedia traits
        try:
            wiki = fetch_wikipedia_traits(session, wiki_query)
        except Exception as e:
            wiki = {"error": str(e), "url": None, "traits": {}, "title": None}

        # Wikibooks Bestimmungsbuch
        try:
            wb = fetch_wikibooks_bestimmungsbuch(session, scientific_name, german_name)
        except Exception as e:
            wb = {"error": str(e), "url": None, "title": None}

        identification_traits = wiki.get("traits", {}) if isinstance(wiki, dict) else {}

        determination_key = None
        if isinstance(wb, dict) and wb.get("extracted") and not wb.get("error"):
            determination_key = {
                "source": "Wikibooks Bestimmungsbuch Pflanzen Mitteleuropas",
                "title": wb.get("title"),
                "url": wb.get("url"),
                "method": wb.get("method"),
                "score": wb.get("score"),
                "intro": (wb.get("extracted") or {}).get("intro", []),
                "matching_snippets": (wb.get("extracted") or {}).get("matching_snippets", []),
            }

        wikipedia_source_meta = {}
        if isinstance(wiki, dict):
            wikipedia_source_meta = {
                "title": wiki.get("title"),
                "url": wiki.get("url"),
                "license": wiki.get("license"),
                "attribution_minimum": wiki.get("attribution_minimum"),
                "query_used": wiki_query,
            }
            if wiki.get("error"):
                wikipedia_source_meta["error"] = wiki.get("error")

        wikibooks_source_meta = {}
        if isinstance(wb, dict):
            wikibooks_source_meta = {
                "title": wb.get("title"),
                "url": wb.get("url"),
                "license": wb.get("license"),
                "attribution_minimum": wb.get("attribution_minimum"),
                "method": wb.get("method"),
            }
            if wb.get("error"):
                wikibooks_source_meta["error"] = wb.get("error")

        results.append({
            "scientific_name": scientific_name,
            "name_de": german_name,
            "access_date": today,
            "identification_traits": identification_traits,
            "determination_key": determination_key,
            "sources": {
                "wikipedia_de": wikipedia_source_meta,
                "wikibooks_bestimmungsbuch": wikibooks_source_meta,
            }
        })

        time.sleep(delay)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", default="names_de.json", help="Path to names_de.json mapping scientific -> German name")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests in seconds")
    parser.add_argument(
        "--prefer-scientific-for-wikipedia",
        action="store_true",
        help="Use the scientific name for Wikipedia lookup instead of the German name",
    )
    args = parser.parse_args()

    process(
        names_json_path=args.names,
        output_json=args.output,
        delay=args.delay,
        prefer_german_name_for_wikipedia=not args.prefer_scientific_for_wikipedia,
    )