
"""
French Corpus Builder for LLM Pretraining.

This script downloads, cleans and structures a French text corpus
for training both:
    - the Byte-Level BPE tokenizer
    - the transformer LLM (~10M parameters)

It produces two outputs:
    1. corpus.txt          -> full concatenated corpus
    2. corpus_docs/        -> one file per document (for tokenizer training)
    
Author: Robin (+ help from AI)
Date:   December 2025
"""

import os
import re
import random
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from typing import List, Tuple


# =========================
# Configuration
# =========================

MAX_WIKI_PAGES = 1500
MIN_CHARS_PER_PAGE = 800  # reject very short pages

DOC_SEPARATOR = "<|doc|>"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

OUTPUT_DIR = "corpus_docs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Base Wikipedia pages
BASE_WIKI_URLS = [
    "https://fr.wikipedia.org/wiki/Science",
    "https://fr.wikipedia.org/wiki/Philosophie",
    "https://fr.wikipedia.org/wiki/Mathématiques",
    "https://fr.wikipedia.org/wiki/Technologie",
    "https://fr.wikipedia.org/wiki/Intelligence_artificielle",
    "https://fr.wikipedia.org/wiki/Langage_naturel",
    "https://fr.wikipedia.org/wiki/Apprentissage_automatique",
    "https://fr.wikipedia.org/wiki/Réseau_neuronal_artificiel",
    "https://fr.wikipedia.org/wiki/Apprentissage_profond",
    "https://fr.wikipedia.org/wiki/Statistiques",
    "https://fr.wikipedia.org/wiki/Linguistique_de_corpus",
    "https://fr.wikipedia.org/wiki/Littérature_française",
    "https://fr.wikipedia.org/wiki/Histoire_de_France",
    "https://fr.wikipedia.org/wiki/Économie",
    "https://fr.wikipedia.org/wiki/Psychologie",
    "https://fr.wikipedia.org/wiki/Sociologie",
    "https://fr.wikipedia.org/wiki/Politique_en_France",
    "https://fr.wikipedia.org/wiki/Droit_en_France",
    "https://fr.wikipedia.org/wiki/Phénoménologie",
    "https://fr.wikipedia.org/wiki/Physique",
    "https://fr.wikipedia.org/wiki/Biologie",
    "https://fr.wikipedia.org/wiki/Code_civil_français",
    "https://fr.wikipedia.org/wiki/Constitution_française_du_4_octobre_1958",
]



# Theatre
THEATRE_URLS = [
    ("Le Cid — Corneille", "https://www.theatre-classique.fr/pages/documents/CORNEILLEP_CID.xml"),
    ("Le Médecin malgré lui — Molière", "https://www.theatre-classique.fr/pages/documents/MOLIERE_MEDECINMALGRELUI.xml"),
    ("Lorenzaccio — Musset", "https://www.theatre-classique.fr/pages/documents/MUSSET_LORENZACCIO.xml"),
    ("Racine — Phèdre", "https://www.theatre-classique.fr/pages/documents/RACINE_PHEDRE.xml")
]


# CFPP2000 
CFPP_URLS = [
    ("CFPP — Rosemonde & Patrick (7e)",
     "http://cfpp2000.univ-paris3.fr/data/public/7eme/Rosemonde_Ehrard_60_et_Patrick_Bernard_49_7e/Rosemonde_Ehrard_60_et_Patrick_Bernard_49_7e-html-v2.xml"),

    ("CFPP — Paul & Pierre-Marie (18e)",
     "http://cfpp2000.univ-paris3.fr/data/public/18eme/Paul_Simo_20_Pierre_Marie-Simo_M_34-18e/Paul_Simo_20_Pierre_Marie-Simo_M_34_18e-html-v2.xml"),
]


# Gutenberg texts
GUTENBERG_URLS = [
    ("Jules Verne — De la Terre à la Lune", "https://www.gutenberg.org/cache/epub/799/pg799.txt"),
    ("Alexandre Dumas — Les Trois Mousquetaires", "https://www.gutenberg.org/cache/epub/13951/pg13951.txt"),
    ("Victor Hugo — Les Misérables (Tome 1)", "https://www.gutenberg.org/cache/epub/13951/pg13951.txt"),
    ("Émile Zola — Germinal", "https://www.gutenberg.org/cache/epub/5711/pg5711.txt"),
    ("Flaubert — Madame Bovary", "https://www.gutenberg.org/cache/epub/14155/pg14155.txt"),
    ("Maupassant — Bel-Ami", "https://www.gutenberg.org/cache/epub/3733/pg3733.txt"),
]




# =========================
# Utilities
# =========================

def fetch(url: str, encoding: str | None = None) -> str:
    """Download a URL with proper headers."""
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    if encoding:
        r.encoding = encoding
    return r.text


# =========================
# Text Cleaning
# =========================

def clean_text(text):
    """ Normalize whitespace and punctuation spacing."""

    # Remove spaces before punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    # Remove spaces after apostrophes 
    text = re.sub(r"’\s+", "’", text)

    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove spaces before parentheses or quotes
    text = re.sub(r"\s+([«»()])", r"\1", text)

    return text.strip()


# =========================
# Wikipedia extraction
# =========================

def extract_wikipedia_links(url: str) -> List[str]:
    """ Extract all internal Wikipedia links from a page. """
    html = fetch(url)
    soup = BeautifulSoup(html, "html.parser")

    content = soup.find("div", {"id": "mw-content-text"})
    if not content:
        return []

    article = content.find("div", class_="mw-parser-output")
    if not article:
        return []

    links = []
    for a in article.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/wiki/") and ":" not in href:
            links.append("https://fr.wikipedia.org" + href)

    return list(dict.fromkeys(links))


def extract_wikipedia_article(url):
    """ Clean and extract readable text from a Wikipedia article. """
    html = fetch(url)
    soup = BeautifulSoup(html, "html.parser")

    content = soup.find("div", {"id": "mw-content-text"})
    if not content:
        return ""

    article = content.find("div", class_="mw-parser-output")
    if not article:
        return ""

    # Remove useless blocks
    for selector in [
        "table.infobox", "table.navbox", "table.vertical-navbox",
        "div.reflist", "div.navbox", "div.sidebar", "div.toc",
        "sup", "style"
    ]:
        for tag in article.select(selector):
            tag.decompose()

    lines = []
    for elem in article.find_all(["h1", "h2", "h3", "p"]):
        text = elem.get_text(" ", strip=True)
        if not text:
            continue

        if elem.name == "h1":
            lines.append(f"# {text}")
        elif elem.name == "h2":
            lines.append(f"## {text}")
        elif elem.name == "h3":
            lines.append(f"### {text}")
        else:
            lines.append(text)

    txt = "\n".join(lines)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


# =========================
# Theatre extraction
# ========================= 

def extract_theatre(url):
    """ Extract structured dialogue from theatre XML. """
    xml = fetch(url)
    root = ET.fromstring(xml)

    lines = []

    for act in root.findall(".//div1[@type='act']"):
        act_title = act.find("head")
        if act_title is not None:
            lines.append(f"# {act_title.text.strip()}")

        for scene in act.findall("div2[@type='scene']"):
            scene_title = scene.find("head")
            if scene_title is not None:
                lines.append(f"## {scene_title.text.strip()}")

            for sp in scene.findall("sp"):
                speaker_tag = sp.find("speaker")
                if speaker_tag is None:
                    continue

                speaker = speaker_tag.text.strip().replace(":", "").capitalize()

                # Vers <l>
                for l in sp.findall("l"):
                    text = "".join(l.itertext()).strip()
                    text = re.sub(r"^\d+\s+", "", text)
                    text = re.sub(r"\([^)]*\)", "", text)
                    if text:
                        lines.append(f"{speaker}: {text}")

                # Prose <p><s>
                for p in sp.findall("p"):
                    for s in p.findall("s"):
                        text = "".join(s.itertext()).strip()
                        text = re.sub(r"^\d+\s+", "", text)
                        text = re.sub(r"\([^)]*\)", "", text)
                        if text:
                            lines.append(f"{speaker}: {text}")

    return "\n".join(lines).strip()


# =========================
# CFPP extraction
# =========================

def extract_cfpp(url):
    """ Extract spoken French dialogue from CFPP2000 XML. """
    xml = fetch(url, encoding="ISO-8859-1")
    root = ET.fromstring(xml)

    lines = []

    for turn in root.findall(".//Turn"):
        speaker = (turn.get("speaker") or "").strip()
        if not speaker:
            continue

        text = "".join(turn.itertext()).strip()

        # Remove metadata
        for child in turn:
            if child.tag in ("Sync", "Who", "Comment"):
                t = "".join(child.itertext()).strip()
                text = text.replace(t, "")

        text = text.strip()
        if text:
            lines.append(f"{speaker}: {text}")

    return "\n".join(lines).strip()


# =========================
# Gutenberg extraction
# =========================

def extract_gutenberg(url):
    """ Extract plain text from Gutenberg .txt files. """
    txt = fetch(url)
    txt = txt.replace("\r", "")

    # Remove Gutenberg header/footer
    txt = re.sub(r"\*\*\* START OF.*?\*\*\*", "", txt, flags=re.DOTALL)
    txt = re.sub(r"\*\*\* END OF.*?\*\*\*", "", txt, flags=re.DOTALL)

    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)

    return txt.strip()


# =========================
# Corpus Builder
# =========================

def build_corpus():
    docs = []

    # ------------------------------
    # Wikipedia expansion
    # ------------------------------
    all_links = set()
    for url in BASE_WIKI_URLS:
        try:
            linked = extract_wikipedia_links(url)
            all_links.update(linked)
        except:
            pass

    all_links = list(all_links)
    random.shuffle(all_links)

    limited_links = all_links[:MAX_WIKI_PAGES]
    wiki_urls = BASE_WIKI_URLS + limited_links

    print(f"Scraping {len(wiki_urls)} Wikipedia pages...")

    for url in wiki_urls:
        try:
            txt = extract_wikipedia_article(url)
            if len(txt) >= MIN_CHARS_PER_PAGE:
                docs.append(clean_text(txt))
        except:
            pass

    # ------------------------------
    # Theatre
    # ------------------------------
    for label, url in THEATRE_URLS:
        try:
            txt = extract_theatre(url)
            docs.append(clean_text(f"# {label}\n{txt}"))
        except:
            pass

    # ------------------------------
    # CFPP
    # ------------------------------
    for label, url in CFPP_URLS:
        try:
            txt = extract_cfpp(url)
            docs.append(clean_text(f"# {label}\n{txt}"))
        except:
            pass

    # ------------------------------
    # Gutenberg
    # ------------------------------
    for label, url in GUTENBERG_URLS:
        try:
            txt = extract_gutenberg(url)
            docs.append(clean_text(f"# {label}\n{txt}"))
        except:
            pass

    # ------------------------------
    # Save per-document files
    # ------------------------------
    for i, doc in enumerate(docs):
        path = os.path.join(OUTPUT_DIR, f"doc_{i:05d}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc)

    # ------------------------------
    # Save concatenated corpus
    # ------------------------------
    full = f"\n\n{DOC_SEPARATOR}\n\n".join(docs)
    with open("corpus.txt", "w", encoding="utf-8") as f:
        f.write(full)

    print(f"Done. {len(docs)} documents saved.")
    print(f"Total characters: {len(full)}")


if __name__ == "__main__":
    build_corpus()