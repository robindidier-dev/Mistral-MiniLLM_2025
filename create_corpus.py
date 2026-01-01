
"""
French Corpus Builder for LLM Pretraining.

This script downloads, cleans and structures a French text corpus
for training both:
    - the Byte-Level BPE tokenizer
    - the transformer LLM (~10M parameters)

It produces two outputs:
    1. corpus.txt          -> full concatenated corpus (not used anymore)
    2. corpus_docs/        -> one file per document (for tokenizer and model training)
    
Author: Robin (+ help from AI)
Date:   December 2025
"""

import os
import re
import unicodedata
import random
import requests
import urllib.parse
import xml.etree.ElementTree as ET

# =========================
# Configuration générale
# =========================

MAX_WIKI_PAGES = 1500
MIN_CHARS_PER_PAGE = 800
DOC_SEPARATOR = "<|doc|>"

OUTPUT_DIR = "corpus_docs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64 "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# =========================
# Espacement des titres Markdown
# =========================

def add_spacing_around_markdown_headers(text: str) -> str:
    """
    Ajoute des sauts de ligne autour des titres Markdown et dialogues :
    - Triple saut de ligne AVANT les titres (# ## ### **) et dialogues ([...])
    - Simple saut de ligne APRÈS les titres et dialogues
    """
    lines = text.split("\n")
    result = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Détecte les titres Markdown : # ## ### ou **...** en début de ligne
        is_header = (
            stripped.startswith("#") or 
            (stripped.startswith("**") and stripped.endswith("**"))
        )
        
        # Détecte les dialogues : [...]
        is_dialogue = stripped.startswith("[")
        
        if is_header or is_dialogue:
            # Ajouter deux sauts de ligne avant (s'il n'y en a pas déjà)
            # Supprimer les lignes vides à la fin
            while result and result[-1].strip() == "":
                result.pop()
            
            # Ajouter deux lignes vides de séparation
            if result:  # S'il y a déjà du contenu
                result.append("")
                result.append("")
            
            result.append(stripped)
            result.append("")  # Ajouter une ligne vide après
        else:
            # Pour les autres lignes, les garder telles quelles
            if stripped:  # Ne pas ajouter les lignes vides
                result.append(stripped)
            elif result and result[-1] != "":  # Préserver une ligne vide si elle n'est pas redondante
                result.append("")
    
    # Rejoindre
    text = "\n".join(result)
    # Réduire les sauts de ligne 4+ à triple
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    
    return text.strip()

# =========================
# Normalisation de base
# =========================

def normalize_unicode_base(text: str) -> str:
    """
    Normalisation Unicode de base :
      - NFKC
      - suppression des caractères de contrôle (sauf \\n)
      - conversion des espaces Unicode en espace standard
      - suppression des caractères de largeur zéro
    """
    # NFKC
    text = unicodedata.normalize("NFKC", text)

    # Supprimer les caractères de contrôle (gardez \\n)
    out_chars: list[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch == "\n":
            out_chars.append(ch)
        elif cat[0] == "C":
            # C* = Other (control, format, etc.)
            continue
        else:
            out_chars.append(ch)
    text = "".join(out_chars)

    # Espaces Unicode -> espace normal
    text = re.sub(r"[\u00A0\u202F\u2009\u2002\u2003\u2007\u2060]", " ", text)
    # Supprimer zero-width / BOM
    text = re.sub(r"[\u200B\u200C\u200D\uFEFF]", "", text)

    return text

# =========================
# Guillemets & apostrophes
# =========================

def normalize_quotes_and_apostrophes(text: str) -> str:
    """
    - Convertit les guillemets anglais " et ' utilisés comme guillemets en « »
    - Convertit toutes les apostrophes droites ' en apostrophes typographiques ’
    - Assure des espaces autour de « et »
    Hypothèse : on préfère simplifier pour un petit modèle francophone,
    quitte à être un peu agressif.
    """
    # 1) Apostrophe droite -> apostrophe typographique
    # Attention : on le fait d'abord, pour que les remplacements de guillemets
    # sur '...'/"...' s'appliquent ensuite sur du texte déjà stable.
    text = text.replace("'", "’")

    # 2) Guillemets anglais -> guillemets français
    # Cas simples : "contenu"
    text = re.sub(r'"([^"\n]+)"', r'« \1 »', text)

    # Par sécurité, on remplace les guillemets restants isolés
    # "mot -> « mot
    text = re.sub(r'"(\S)', r'« \1', text)
    # mot" -> mot »
    text = re.sub(r'(\S)"', r'\1 »', text)
    # Si jamais il reste un " seul, on le supprime
    text = text.replace('"', "")

    # 3) Assurer les espaces autour de « et »
    # Espace avant « si manquant
    text = re.sub(r"(\S)«", r"\1 «", text)
    # Espace après « si manquant
    text = re.sub(r"«(\S)", r"« \1", text)
    # Espace avant » si manquant
    text = re.sub(r"(\S)»", r"\1 »", text)
    # Espace après » si manquant
    text = re.sub(r"»(\S)", r"» \1", text)

    # Réduire les doubles espaces générés
    text = re.sub(r"[ ]{2,}", " ", text)

    return text

# =========================
# Nettoyage de texte générique
# =========================

def clean_text_generic(text: str) -> str:
    """
    Nettoyage minimal :
      - normalisation des fins de ligne
      - réduction des espaces multiples
      - réduction des sauts de ligne multiples
      - conversion des sauts de ligne simples en espaces (pas les doubles)
    On ne touche pas à la ponctuation, juste à la structure.
    """
    text = text.replace("\r", "\n")

    # Espaces horizontaux multiples -> un espace
    text = re.sub(r"[ \t]+", " ", text)

    # Sauts de ligne multiples (3+) -> double saut (séparateur de paragraphe)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Sauts de ligne simples -> espaces (reconstitue les paragraphes sur une ligne)
    # MAIS préserver les doubles sauts de ligne
    # Remplacer \n par espace SAUF si c'est \n\n
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    
    # Nettoyer les espaces en début/fin de lignes
    lines = text.split("\n")
    lines = [line.strip() for line in lines]
    text = "\n".join(lines)

    return text.strip()

# =========================
# Normalisation complète utilisée partout
# =========================

def normalize_full(text: str) -> str:
    """
    Pipeline de normalisation standard pour tout le corpus :
      1. Normalisation Unicode de base
      2. Normalisation guillemets & apostrophes
      3. Espacement autour des titres Markdown et dialogues
      4. Nettoyage générique
    """
    text = normalize_unicode_base(text)
    text = normalize_quotes_and_apostrophes(text)
    text = add_spacing_around_markdown_headers(text)
    text = clean_text_generic(text)
    return text




def collapse_paragraph_breaks_gutenberg(text: str) -> str:
    """
    Nettoie le texte Gutenberg en se basant sur la grammaire pour fusionner 
    les paragraphes coupés par erreur (OCR ou formatage ancien).
    
    Logique :
    - Supprime d'abord les retours à la ligne simples (hard-wrap).
    - Analyse les doubles retours à la ligne (\n\n).
    - Si un bloc ne finit pas par une ponctuation forte ou si le suivant commence
      par une minuscule (sans être un dialogue), on fusionne.
    """
    # 1. Normalisation des sauts de ligne
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # 2. Suppression des sauts de ligne simples (remplacement par espace)
    # Cela transforme les paragraphes "physiques" (lignes de 80 chars) en une seule ligne
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # 3. Découpage en blocs candidats (sur les doubles sauts restants)
    blocks = [b.strip() for b in text.split('\n\n') if b.strip()]
    
    merged_blocks = []
    
    # Ponctuations marquant une VRAIE fin de paragraphe
    END_PUNCTUATION = ('.', '?', '!', ':', '»', '"', '…', '”')
    
    for block in blocks:
        if not merged_blocks:
            merged_blocks.append(block)
            continue
            
        last_block = merged_blocks[-1]
        
        # --- ANALYSE GRAMMATICALE ---
        
        # Le bloc précédent finit-il "en suspens" ? (Pas de point, ?, !)
        ends_abruptly = not last_block.endswith(END_PUNCTUATION)
        
        # Le bloc actuel commence-t-il comme une suite ? (Minuscule)
        starts_lowercase = block[0].islower() if block else False
        
        # Le bloc actuel est-il un dialogue ? (Commence par tiret)
        is_dialogue = block.startswith(('-', '—', '–'))

        # CRITÈRE DE FUSION :
        # On fusionne si le précédent n'est pas fini OU si l'actuel est une suite évidente
        # MAIS on ne fusionne jamais si l'actuel marque le début d'un dialogue.
        if (ends_abruptly or starts_lowercase) and not is_dialogue:
            # Gestion de l'espace de fusion (éviter " l' homme " -> "l'homme")
            separator = " "
            if last_block.endswith("'") or last_block.endswith("’"):
                separator = ""
            
            merged_blocks[-1] = last_block + separator + block
        else:
            # C'est un nouveau paragraphe légitime
            merged_blocks.append(block)

    # 4. Reconstruction du texte complet
    full_text = "\n\n".join(merged_blocks)
    
    # 5. Nettoyage final des espaces multiples (créés par les fusions)
    full_text = re.sub(r'[ \t]+', ' ', full_text)
    
    # 6. (Optionnel) Ajout d'un saut de ligne propre avant les dialogues mal collés
    # Exemple : "Il dit. — Bonjour" -> "Il dit.\n\n— Bonjour"
    full_text = re.sub(r'(?<=[\.\!\?»])\s+(—|--|-)\s+', r'\n\n— ', full_text)
    
    return full_text

def normalize_full_gutenberg(text: str) -> str:
    """
    Pipeline de normalisation pour Gutenberg (sans espacement supplémentaire des dialogues) :
      1. Normalisation Unicode de base
      2. Normalisation guillemets & apostrophes
      3. Nettoyage générique (sans add_spacing_around_markdown_headers)
      4. Collapsing des \n\n entre paragraphes de texte simple
    """
    text = normalize_unicode_base(text)
    text = normalize_quotes_and_apostrophes(text)
    text = clean_text_generic(text)
    text = collapse_paragraph_breaks_gutenberg(text)
    return text

# =========================
# Helper HTTP générique
# =========================

def fetch(url: str, encoding: str | None = None) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    if encoding:
        resp.encoding = encoding
    return resp.text

# =========================
# Bloc 2 — Extraction Wikipédia via API
# =========================

def crawl_wikipedia(start_titles: list[str], max_scrapped: int) -> list[str]:
    """
    Explore Wikipédia en largeur :
      - part des titres initiaux
      - récupère les liens internes via l'API
      - évite les doublons
      - s'arrête à max_scrapped pages valides
    Retourne une liste de documents nettoyés.
    """

    visited = set()
    queue = list(start_titles)
    docs = []

    while queue and len(docs) < max_scrapped:
        title = queue.pop(0)

        # Éviter les doublons
        if title in visited:
            continue
        visited.add(title)

        # Récupérer et nettoyer la page
        print(f"[Wiki] Traitement: {title}...")
        doc = process_wikipedia_page(title)
        if not doc:
            print(f"  → Rejeté (contenu insuffisant ou erreur)")
            continue

        docs.append(doc)
        print(f"  → Accepté ({len(doc)} chars)")

        # Extraire les liens internes via l'API
        try:
            links = fetch_wiki_links(title)
            print(f"  → {len(links)} liens trouvés")
        except Exception as e:
            print(f"  → Erreur liens: {e}")
            continue

        for link in links:
            # Filtrer les namespaces non pertinents
            if any(link.startswith(prefix) for prefix in ["Fichier:", "Catégorie:", "Aide:", "Portail:", "Spécial:"]):
                continue

            # Ajouter à la queue si pas encore visité
            if link not in visited:
                queue.append(link)

    return docs


WIKI_API = "https://fr.wikipedia.org/w/api.php"

# Sections à supprimer (tout ce qui suit est ignoré)
WIKI_CUTOFF_SECTIONS = [
    "Notes et références",
    "Voir aussi",
    "Bibliographie",
    "Articles connexes",
    "Liens externes",
    "Références",
    "Sources",
    "Annexes",
    "Portail",
]

def url_to_title(url: str) -> str:
    """Convertit une URL fr.wikipedia en titre d’article."""
    name = url.rsplit("/", 1)[-1]
    name = urllib.parse.unquote(name)
    return name.replace("_", " ")

def fetch_wiki_extract(title: str) -> str:
    """
    Récupère le texte brut d’un article Wikipédia via l’API (explaintext).
    Pas de HTML, pas de balises, pas de scripts.
    """
    params = {
        "action": "query",
        "prop": "extracts",
        "titles": title,
        "redirects": 1,
        "format": "json",
        "formatversion": 2,
        "explaintext": 1,
    }
    resp = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return ""
    page = pages[0]
    if "missing" in page:
        return ""

    return page.get("extract", "") or ""

def fetch_wiki_links(title: str) -> list[str]:
    """
    Récupère les liens internes d’un article (titres d’autres pages).
    """
    links = []
    plcontinue = None

    while True:
        params = {
            "action": "query",
            "titles": title,
            "prop": "links",
            "plnamespace": 0,
            "pllimit": "max",
            "format": "json",
            "formatversion": 2,
        }
        if plcontinue:
            params["plcontinue"] = plcontinue

        resp = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        pages = data.get("query", {}).get("pages", [])
        if not pages:
            break

        page = pages[0]
        for link in page.get("links", []):
            t = link.get("title")
            if t:
                links.append(t)

        cont = data.get("continue")
        if not cont or "plcontinue" not in cont:
            break

        plcontinue = cont["plcontinue"]

    # Déduplication en conservant l’ordre
    seen = set()
    out = []
    for t in links:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

# =========================
# Nettoyage Wikipédia
# =========================

def convert_wiki_headings_to_markdown(text: str) -> str:
    # Niveau 4 : ==== Sous-sous-section ====  → **Sous-sous-section**
    text = re.sub(
        r"^====\s*(.*?)\s*====\s*$",
        r"**\1**",
        text,
        flags=re.MULTILINE
    )

    # Niveau 3 : === Sous-section === → ### Sous-section
    text = re.sub(
        r"^===\s*(.*?)\s*===\s*$",
        r"### \1",
        text,
        flags=re.MULTILINE
    )

    # Niveau 2 : == Section == → ## Section
    text = re.sub(
        r"^==\s*(.*?)\s*==\s*$",
        r"## \1",
        text,
        flags=re.MULTILINE
    )

    # Niveau 1 : = Titre = → # Titre
    text = re.sub(
        r"^=\s*(.*?)\s*=\s*$",
        r"# \1",
        text,
        flags=re.MULTILINE
    )

    # Nettoyage des = résiduels dans les titres Markdown
    text = re.sub(
        r"^(###|##|#)\s*=\s*(.*?)\s*=\s*$",
        r"\1 \2",
        text,
        flags=re.MULTILINE
    )

    return text


def clean_wikipedia_text(text: str) -> str:
    # 1. Nettoyage préliminaire (homonymies, redirections, etc.)
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        stripped = line.strip()

        if re.match(r"^Pour l[’']?article homophone", stripped, flags=re.IGNORECASE):
            continue
        if re.match(r"^Pour les articles homonymes", stripped, flags=re.IGNORECASE):
            continue
        if re.match(r".*redirige ici.*", stripped, flags=re.IGNORECASE):
            continue
        if re.match(r"^Cet article est une ébauche", stripped, flags=re.IGNORECASE):
            continue

        cleaned.append(line)

    text = "\n".join(cleaned)

    # 2. Conversion Markdown
    text = convert_wiki_headings_to_markdown(text)

    # Couper après les sections parasites
    final_lines = []
    for line in text.splitlines():
        stripped = line.strip()

        if any(
            stripped.startswith(f"## {sec}") or stripped.startswith(f"### {sec}") or stripped.startswith(f"**{sec}**")
            for sec in WIKI_CUTOFF_SECTIONS
        ):
            break

        final_lines.append(line)

    text = "\n".join(final_lines).strip()

    return text


# =========================
# Pipeline Wikipédia complet
# =========================

def process_wikipedia_page(title: str) -> str | None:
    """
    Pipeline complet :
      1. fetch API
      2. nettoyage Wikipédia
      3. normalisation Unicode + guillemets + apostrophes
      4. nettoyage générique
      5. encapsulation dans un document Markdown
    """
    raw = fetch_wiki_extract(title)
    if not raw or len(raw) < MIN_CHARS_PER_PAGE:
        return None

    cleaned = clean_wikipedia_text(raw)
    normalized = normalize_full(cleaned)

    if not normalized:
        return None

    return f"# {title}\n\n{normalized}"


# =========================
# Bloc 3 — Théâtre (TEI XML)
# =========================

def extract_theatre(url: str) -> str:
    """
    Extraction d’une pièce de théâtre au format TEI XML depuis theatre-classique.fr.
    Structure finale :
        # Titre
        ## ACTE I
        ### SCÈNE PREMIÈRE
        [Personnage] texte
        [Personnage] texte
        ...
    """
    xml = fetch(url)
    root = ET.fromstring(xml)

    lines: list[str] = []

    # Titre principal (si présent dans <title>)
    title_tag = root.find(".//title")
    if title_tag is not None and title_tag.text:
        title = title_tag.text.strip()
        lines.append(f"# {title}\n\n")
    else:
        lines.append("# Pièce inconnue\n\n")

    # Parcours des actes
    for act in root.findall(".//div1[@type='act']"):
        act_title = act.find("head")
        if act_title is not None and act_title.text:
            lines.append(f"## {act_title.text.strip()}\n\n")

        # Parcours des scènes
        for scene in act.findall("div2[@type='scene']"):
            scene_title = scene.find("head")
            if scene_title is not None and scene_title.text:
                lines.append(f"### {scene_title.text.strip()}\n\n")

            # Parcours des répliques <sp>
            for sp in scene.findall("sp"):
                speaker_tag = sp.find("speaker")
                if speaker_tag is None or speaker_tag.text is None:
                    continue

                speaker = speaker_tag.text.strip().replace(":", "")
                speaker = speaker.capitalize()

                # Répliques en vers <l>
                for l in sp.findall("l"):
                    text = "".join(l.itertext()).strip()
                    text = re.sub(r"^\d+\s+", "", text)  # numéros de vers
                    text = re.sub(r"\([^)]*\)", "", text)  # didascalies
                    text = text.strip()
                    if text:
                        lines.append(f"[{speaker}] {text}\n")

                # Répliques en prose <p><s>
                for p in sp.findall("p"):
                    for s in p.findall("s"):
                        text = "".join(s.itertext()).strip()
                        text = re.sub(r"^\d+\s+", "", text)
                        text = re.sub(r"\([^)]*\)", "", text)
                        text = text.strip()
                        if text:
                            lines.append(f"[{speaker}] {text}\n")

            lines.append("\n")

    # Fusion finale
    raw_text = "".join(lines)

    # Normalisation complète (Bloc 1)
    normalized = normalize_full(raw_text)

    return normalized.strip()


# =========================
# Bloc 5 — Gutenberg (romans) — Version corrigée
# =========================

def strip_gutenberg_boilerplate(text: str) -> str:
    """
    Supprime TOUT le boilerplate Gutenberg (Header, Footer, Licences, ASCII art).
    """
    lines = text.splitlines()
    cleaned = []
    inside_text = False

    # Marqueurs de début et fin courants chez Gutenberg
    START_MARKERS = [r"\*\*\* START", r"\*\*\*START", r"START OF THE PROJECT", r"START OF THIS PROJECT"]
    END_MARKERS = [r"\*\*\* END", r"\*\*\*END", r"END OF THE PROJECT", r"END OF THIS PROJECT"]

    for line in lines:
        stripped = line.strip()

        # Détection du début
        if not inside_text:
            if any(re.search(m, stripped, flags=re.IGNORECASE) for m in START_MARKERS):
                inside_text = True
            continue # On ignore tout tant qu'on n'a pas trouvé le start

        # Détection de la fin
        if any(re.search(m, stripped, flags=re.IGNORECASE) for m in END_MARKERS):
            break

        # Nettoyage interne (Lignes de pollution courantes)
        if re.search(r"Project Gutenberg", stripped, flags=re.IGNORECASE): continue
        if re.search(r"www\.gutenberg\.org", stripped, flags=re.IGNORECASE): continue
        if re.match(r"^[-=_*]{3,}$", stripped): continue # Séparateurs ASCII

        cleaned.append(line)

    # Si les marqueurs ont échoué (cas rares), on rend le texte brut nettoyé au mieux
    result = "\n".join(cleaned).strip()
    return result if result else text


def standardize_dialogue_block(block: str) -> str:
    """
    Applique la standardisation des dialogues sur un bloc (paragraphe) entier.
    - Transforme 'Nom. —' en '[Nom]'
    - Transforme les tirets (--, —, -) en tiret demi-cadratin (–)
    """
    # Cible : Demi-cadratin
    TARGET_DASH = "–"
    DASH_PATTERN = r"(?:--|—|–|-)"

    # 1. Cas : Nom. — texte  => [Nom] texte
    # On cherche un motif au DÉBUT du bloc
    match_named = re.match(r"^([A-ZÉÈÀÂÎÔÛ][A-Za-zÉÈÀÂÎÔÛéèàâîôûç\s\-']+)\.\s*" + DASH_PATTERN + r"\s*(.*)$", block, flags=re.DOTALL)
    if match_named:
        speaker = match_named.group(1).strip()
        content = match_named.group(2).strip()
        return f"[{speaker}] {content}"

    # 2. Cas : Dialogue anonyme (— texte) => – texte
    match_anon = re.match(r"^" + DASH_PATTERN + r"\s*(.*)$", block, flags=re.DOTALL)
    if match_anon:
        content = match_anon.group(1).strip()
        return f"{TARGET_DASH} {content}"

    return block


def clean_gutenberg_text(text: str) -> str:
    """
    Pipeline complet de nettoyage et reconstruction.
    Remplace l'ancienne logique 'buffer' par la logique 'grammaticale' + gestion dialogues.
    """
    # 1. Suppression du boilerplate
    text = strip_gutenberg_boilerplate(text)

    # 2. Normalisation technique
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # 3. Suppression du hard-wrap (remplace saut de ligne simple par espace)
    # Les vrais paragraphes sont séparés par \n\n. Les autres \n sont juste cosmétiques.
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # 4. Reconstruction des blocs brisés (Logique "Grammar Glue")
    # On découpe sur les doubles sauts de ligne
    raw_blocks = [b.strip() for b in text.split('\n\n') if b.strip()]
    merged_blocks = []
    END_PUNCTUATION = ('.', '?', '!', ':', '»', '"', '…', '”')

    for block in raw_blocks:
        if not merged_blocks:
            merged_blocks.append(block)
            continue
        
        last_block = merged_blocks[-1]
        
        # Critères de fusion :
        # - Le précédent ne finit pas par une ponctuation forte
        # - OU l'actuel commence par une minuscule (et n'est pas un dialogue)
        ends_abruptly = not last_block.endswith(END_PUNCTUATION)
        starts_lowercase = block[0].islower() if block else False
        is_dialogue_start = re.match(r'^\s*(?:--|—|–|-)', block)

        if (ends_abruptly or starts_lowercase) and not is_dialogue_start:
            # Fusion
            sep = " " if not (last_block.endswith("'") or last_block.endswith("’")) else ""
            merged_blocks[-1] = last_block + sep + block
        else:
            merged_blocks.append(block)

    # 5. Réassemblage temporaire
    full_text = "\n\n".join(merged_blocks)
    full_text = re.sub(r'[ \t]+', ' ', full_text) # Nettoyage espaces multiples

    # 6. Éclatement des dialogues imbriqués (ex: "Quoi?--Rien")
    # On insère un saut de ligne et un tiret standard
    TARGET_DASH = "–"
    pattern_split = r'(?<=[.?!»”])\s*(?:--|—|–|-)\s*(?=[A-ZÉÈÀÂÎÔÛ"«])'
    full_text = re.sub(pattern_split, f'\n\n{TARGET_DASH} ', full_text)

    # 7. Standardisation finale bloc par bloc (Tirets et [Noms])
    # On redécoupe le texte propre pour appliquer le formatage des dialogues
    final_blocks = []
    for block in full_text.split('\n\n'):
        if not block.strip(): continue
        # Si c'est un titre (tout majuscule court), on garde format Markdown
        if block.isupper() and len(block.split()) < 10:
             final_blocks.append(f"## {block.title()}")
        else:
            # Sinon, on applique la standardisation des dialogues
            final_blocks.append(standardize_dialogue_block(block))

    return "\n\n".join(final_blocks).strip()


def process_gutenberg_file(path: str) -> str | None:
    """
    Lit, nettoie et formate un fichier Gutenberg.
    """
    if not os.path.exists(path):
        return None
        
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
    except Exception as e:
        print(f"Erreur lecture {path}: {e}")
        return None

    cleaned = clean_gutenberg_text(raw)
    
    # Filtre de qualité minimale
    if not cleaned or len(cleaned) < 500:
        return None

    title = os.path.basename(path).replace(".txt", "").replace("_", " ").title()
    return f"# {title}\n\n{cleaned}"


# =========================
# Bloc 6 — Main builder
# =========================

def save_document(text: str, index: int) -> str:
    filename = f"doc_{index:05d}.txt"
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"<|doc|>\n{text}\n")
    return path


def build_corpus(
    wikipedia_titles: list[str],
    theatre_urls: list[str],
    gutenberg_files: list[str],
    max_docs: int = 5000,
    max_scrapped_wiki: int = 40
):
    """
    Pipeline complet :
      - Wikipédia (avec exploration automatique, limité à max_scrapped_wiki)
      - Théâtre
      - Gutenberg
      - max_docs : limite totale du corpus
      - max_scrapped_wiki : limite pour le scraping Wikipedia uniquement
    """

    all_docs = []
    doc_index = 0

    # =========================
    # 1. Wikipédia (crawler) - limité pour laisser place aux autres sources
    # =========================
    print(f"\n=== PHASE 1: WIKIPEDIA (max {max_scrapped_wiki} docs) ===")
    wiki_docs = crawl_wikipedia(wikipedia_titles, max_scrapped_wiki)
    for doc in wiki_docs:
        if doc_index >= max_docs:
            break
        save_document(doc, doc_index)
        all_docs.append(doc)
        doc_index += 1

    # =========================
    # 2. Théâtre
    # =========================
    print("\n=== PHASE 2: THEATRE ===")
    for url in theatre_urls:
        if doc_index >= max_docs:
            break

        try:
            print(f"[Théâtre] Extraction: {url[-30:]}...")
            doc = extract_theatre(url)
            if doc and len(doc) > 300:
                save_document(doc, doc_index)
                all_docs.append(doc)
                doc_index += 1
                print(f"  → Succès ({len(doc)} chars)")
            else:
                print(f"  → Rejeté (contenu insuffisant)")
        except Exception as e:
            print(f"  → Erreur: {e}")

    # =========================
    # 3. Gutenberg
    # =========================
    print("\n=== PHASE 3: GUTENBERG ===")
    for path in gutenberg_files:
        if doc_index >= max_docs:
            break

        print(f"[Gutenberg] Traitement: {os.path.basename(path)}...")
        doc = process_gutenberg_file(path)
        if doc and len(doc) > 500:
            save_document(doc, doc_index)
            all_docs.append(doc)
            doc_index += 1
            print(f"  → Succès ({len(doc)} chars)")
        else:
            print(f"  → Rejeté (contenu insuffisant)")

    # =========================
    # 4. Corpus final concaténé
    # =========================
    final_corpus = "\n".join(f"<|doc|>\n{doc}" for doc in all_docs)
    with open("corpus.txt", "w", encoding="utf-8") as f:
        f.write(final_corpus)

    print(f"Corpus construit : {doc_index} documents.")
    print("Fichiers individuels dans corpus_docs/")
    print("Corpus complet dans corpus.txt")



# =========================
# MAIN — Construction du corpus complet
# =========================

if __name__ == "__main__":

    # ----------------------------------------
    # 1. Wikipédia : convertir URLs → TITRES
    # ----------------------------------------
# ----------------------------------------
    # 4. Wikipedia (Sujets variés pour vocabulaire global)
    # ----------------------------------------
    BASE_WIKI_URLS = [
        # --- Piliers Académiques (Gardés mais réduits) ---
        "https://fr.wikipedia.org/wiki/Science",
        "https://fr.wikipedia.org/wiki/Philosophie",
        "https://fr.wikipedia.org/wiki/Histoire_de_France",
        "https://fr.wikipedia.org/wiki/Littérature_française",
        
        # --- Technologie & Moderne (Condensé pour éviter les doublons) ---
        "https://fr.wikipedia.org/wiki/Intelligence_artificielle", # Suffit pour couvrir ML/Deep Learning
        "https://fr.wikipedia.org/wiki/Internet",                  # Vocabulaire du web, réseaux, com
        "https://fr.wikipedia.org/wiki/Jeu_vidéo",                 # Vocabulaire ludique, moderne, UI
        
        # --- Vie Quotidienne & Culture (CRUCIAL pour un LLM) ---
        "https://fr.wikipedia.org/wiki/Gastronomie_française",      # Vocabulaire nourriture, goût, terroir
        "https://fr.wikipedia.org/wiki/Cuisine",                   # Techniques, verbes d'action (couper, cuire)
        "https://fr.wikipedia.org/wiki/Sport",                     # Verbes de mouvement, compétition, règles
        "https://fr.wikipedia.org/wiki/Mode_(habillement)",        # Vêtements, descriptions physiques, styles
        "https://fr.wikipedia.org/wiki/Cinéma",                    # Arts visuels, narration, critique
        "https://fr.wikipedia.org/wiki/Musique",                   # Sons, instruments, émotions
        
        # --- Nature & Environnement ---
        "https://fr.wikipedia.org/wiki/Biodiversité",              # Animaux, plantes, nature (très riche)
        "https://fr.wikipedia.org/wiki/Climat",                    # Météo, environnement, cataclysmes
        "https://fr.wikipedia.org/wiki/Agriculture",               # Ruralité, outils, production
        
        # --- Société & Humain ---
        "https://fr.wikipedia.org/wiki/Santé",                     # Corps humain, médecine, soins
        "https://fr.wikipedia.org/wiki/Famille",                   # Relations humaines, foyer, éducation
        "https://fr.wikipedia.org/wiki/Économie",                  # Argent, travail, commerce
        "https://fr.wikipedia.org/wiki/Politique",                 # Gouvernance (plus large que juste "France")
        "https://fr.wikipedia.org/wiki/Tourisme",                  # Voyage, géographie, descriptions de lieux
        "https://fr.wikipedia.org/wiki/Architecture",              # Bâtiments, villes, espace
        
        # --- Concepts Abstraits ---
        "https://fr.wikipedia.org/wiki/Temps",                     # Vocabulaire temporel (durée, passé, futur)
        "https://fr.wikipedia.org/wiki/Amour",                     # Sentiments, psychologie relationnelle
    ]

    wikipedia_titles = [url_to_title(u) for u in BASE_WIKI_URLS]

    # ----------------------------------------
    # 2. Théâtre (titres + URLs XML)
    # ----------------------------------------
    
    THEATRE_URLS = [
        # --- Les Classiques (Déjà fonctionnels) ---
        ("Le Cid — Corneille", "https://www.theatre-classique.fr/pages/documents/CORNEILLEP_CID.xml"),
        ("Le Médecin malgré lui — Molière", "https://www.theatre-classique.fr/pages/documents/MOLIERE_MEDECINMALGRELUI.xml"),
        ("Lorenzaccio — Musset", "https://www.theatre-classique.fr/pages/documents/MUSSET_LORENZACCIO.xml"),
        ("Racine — Phèdre", "https://www.theatre-classique.fr/pages/documents/RACINE_PHEDRE.xml"),
        ("Marivaux — Le Jeu de l'amour et du hasard", "https://www.theatre-classique.fr/pages/documents/MARIVAUX_JEUDELAMOURETDUHASARD.xml"),
        
        # --- CORRECTIONS & REMPLACEMENTS (Prose & Textes longs) ---

        # Beaumarchais - Le Barbier de Séville
        # CORRECTION : Le nom du fichier inclut "BARBIERDESEVILLE".
        ("Beaumarchais — Le Barbier de Séville", "https://www.theatre-classique.fr/pages/documents/BEAUMARCHAIS_BARBIERDESEVILLE.xml"),

        # Molière - Le Bourgeois Gentilhomme (5 actes, Prose)
        # REMPLACEMENT de "Mariage de Figaro" (introuvable). Texte très long, vocabulaire varié (musique, philosophie, cuisine).
        ("Molière — Le Bourgeois Gentilhomme", "https://www.theatre-classique.fr/pages/documents/MOLIERE_BOURGEOISGENTILHOMME.xml"),

        # Molière - L'Avare (5 actes, Prose)
        # CONFIRMÉ : Fonctionne et excellent pour les structures de phrases logiques.
        ("Molière — L'Avare", "https://www.theatre-classique.fr/pages/documents/MOLIERE_AVARE.xml"),

        # Molière - Dom Juan (5 actes, Prose)
        # CONFIRMÉ : Fonctionne.
        ("Molière — Dom Juan", "https://www.theatre-classique.fr/pages/documents/MOLIERE_DOMJUAN.xml"),

        # Alfred Jarry - Ubu Roi (1896)
        # REMPLACEMENT de Feydeau. Texte charnière, vocabulaire cru et moderne, prose dynamique.
        ("Alfred Jarry — Ubu Roi", "https://www.theatre-classique.fr/pages/documents/JARRY_UBUROI.xml"),

        # Lesage - Turcaret (5 actes, Prose)
        # AJOUT : Comédie de mœurs sur l'argent. Très utile pour le vocabulaire financier/social.
        ("Lesage — Turcaret", "https://www.theatre-classique.fr/pages/documents/LESAGE_TURCARET.xml"),
    ]

    theatre_urls = [url for (_, url) in THEATRE_URLS]

# ----------------------------------------
    # 3. Gutenberg (titres + URLs TXT)
    #    → Téléchargement local avant traitement
    # ----------------------------------------
    GUTENBERG_URLS = [
        # --- Classiques XIXe (Vocabulaire riche et littéraire) ---
        ("Jules Verne — De la Terre à la Lune", "https://www.gutenberg.org/cache/epub/799/pg799.txt"),
        ("Alexandre Dumas — Les Trois Mousquetaires", "https://www.gutenberg.org/cache/epub/13951/pg13951.txt"),
        ("Victor Hugo — Les Misérables (Tome 1)", "https://www.gutenberg.org/cache/epub/17489/pg17489.txt"),
        ("Émile Zola — Germinal", "https://www.gutenberg.org/cache/epub/5711/pg5711.txt"),
        ("Flaubert — Madame Bovary", "https://www.gutenberg.org/cache/epub/14155/pg14155.txt"),
        ("Maupassant — Bel-Ami", "https://www.gutenberg.org/cache/epub/4647/pg4647.txt"),
        ("Stendhal — Le Rouge et le Noir", "https://www.gutenberg.org/cache/epub/798/pg798.txt"),
        ("Baudelaire — Les Fleurs du Mal", "https://www.gutenberg.org/cache/epub/6099/pg6099.txt"),

        # --- Début XXe siècle (Langue plus moderne, dialogues naturels) ---
        
        # Arsène Lupin (1907) : Style policier, dynamique, argot de l'époque mais structure moderne.
        ("Maurice Leblanc — Arsène Lupin, Gentleman-Cambrioleur", "https://www.gutenberg.org/cache/epub/32854/pg32854.txt"),
        
        # Le Mystère de la Chambre Jaune (1907) : Enquête journalistique, style direct.
        ("Gaston Leroux — Le Mystère de la Chambre Jaune", "https://www.gutenberg.org/cache/epub/13765/pg13765.txt"),
        
        # Le Grand Meaulnes (1913) : Prose poétique mais fluide et simple.
        ("Alain-Fournier — Le Grand Meaulnes", "https://www.gutenberg.org/cache/epub/5781/pg5781.txt"),
        
        # La Guerre des boutons (1912) : CRUCIAL pour un LLM -> contient du langage familier/enfantin et des dialogues vifs.
        ("Louis Pergaud — La Guerre des boutons", "https://www.gutenberg.org/cache/epub/56646/pg56646.txt"),
        
        # Le Diable au corps (1923) : Analyse psychologique, écriture très moderne et incisive.
        ("Raymond Radiguet — Le Diable au corps", "https://www.gutenberg.org/cache/epub/60383/pg60383.txt"),
        
        # Du côté de chez Swann (1913) : Phrases complexes, mais vocabulaire introspectif moderne.
        ("Marcel Proust — Du côté de chez Swann", "https://www.gutenberg.org/cache/epub/2650/pg2650.txt"),
    ]

    gutenberg_files = []
    os.makedirs("gutenberg_downloads", exist_ok=True)

    for title, url in GUTENBERG_URLS:
        filename = "gutenberg_downloads/" + title.replace(" ", "_").replace("—", "_") + ".txt"
        try:
            print(f"[Gutenberg] Téléchargement: {title}...")
            txt = fetch(url)
            with open(filename, "w", encoding="utf-8") as f:
                f.write(txt)
            gutenberg_files.append(filename)
            print(f"  → Succès ({len(txt)} chars)")
        except Exception as e:
            print(f"[WARN] Impossible de télécharger Gutenberg : {title} - {e}")

    # ----------------------------------------
    # 4. Lancement du pipeline complet
    # ----------------------------------------
    build_corpus(
        wikipedia_titles=wikipedia_titles,
        theatre_urls=theatre_urls,
        gutenberg_files=gutenberg_files,
        max_docs=10001,
        max_scrapped_wiki=5000  # Wikis max 40, laisse place pour Théâtre (4) + Gutenberg (6) = 60 total
    )