"""
UD Turkish Treebank'ten Chunk + Clause Verisi Hazirlama
=======================================================
CoNLL-U dosyasindan BIO formatinda CHUNK-OUTER, CHUNK-INNER ve CLAUSE
sutunlarini uretir. PDF ornegindeki 3 sutunlu formati saglar.

Sutunlar:
  CHUNK-OUTER : Dis chunk etiketi (B-NP, I-NP, B-VP, ...)
  CHUNK-INNER : Ic ice gecen yapilar (B-RELCL, I-RELCL, B-COMPCL, ...)
  CLAUSE      : Cumlecik tipi (B-RELCL, I-RELCL, B-COMPCL, I-COMPCL, O)
"""

import sys
import os
import urllib.request
import random
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# ========================================
# 1. UD Turkish Treebank Indir
# ========================================
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

UD_URLS = {
    "train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-BOUN/master/tr_boun-ud-train.conllu",
    "test":  "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-BOUN/master/tr_boun-ud-test.conllu",
    "dev":   "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-BOUN/master/tr_boun-ud-dev.conllu",
}

print("=" * 60)
print("UD Turkce Treebank (BOUN) kontrol ediliyor...")
print("=" * 60)

for split, url in UD_URLS.items():
    out_path = os.path.join(DATA_DIR, f"ud_turkish_{split}.conllu")
    if os.path.exists(out_path):
        print(f"  [{split}] Zaten mevcut.")
        continue
    print(f"  [{split}] Indiriliyor...")
    urllib.request.urlretrieve(url, out_path)
    print(f"  [{split}] OK")

# ========================================
# 2. Chunk + Clause Etiketleme
# ========================================

NP_TAGS   = {"NOUN", "PROPN", "PRON", "DET", "ADJ", "NUM"}
VP_TAGS   = {"VERB", "AUX"}
ADVP_TAGS = {"ADV"}
PP_TAGS   = {"ADP"}

# UD bagimlılık ilişkileri -> Clause tipi
RELCL_DEPRELS  = {"acl:relcl", "acl"}       # goreceli cumlecik
COMPCL_DEPRELS = {"ccomp", "xcomp", "obj"}  # tamamlayici cumlecik
ADVCL_DEPRELS  = {"advcl"}                  # zarfsal cumlecik


def pos_to_chunk(upos):
    if upos in NP_TAGS:   return "NP"
    if upos in VP_TAGS:   return "VP"
    if upos in ADVP_TAGS: return "ADVP"
    if upos in PP_TAGS:   return "PP"
    return "O"


def bio_chunk_sequence(tokens):
    """
    tokens: [(form, upos, deprel, head_idx, token_idx), ...]
    Dis chunk (CHUNK-OUTER) BIO etiketleri uretir.
    """
    labels = []
    prev_chunk = None
    for form, upos, deprel, head, idx in tokens:
        chunk = pos_to_chunk(upos)
        if chunk == "O":
            labels.append("O")
            prev_chunk = None
        elif chunk != prev_chunk:
            labels.append(f"B-{chunk}")
            prev_chunk = chunk
        else:
            labels.append(f"I-{chunk}")
    return labels


def get_subtree(head_idx, children_map):
    """
    Bir dugumun tum alt dugumlerini (subtree) dondurur.
    head_idx: 1-tabanli token index
    """
    result = set()
    stack = [head_idx]
    while stack:
        node = stack.pop()
        result.add(node)
        for child in children_map.get(node, []):
            stack.append(child)
    return result


def compute_clause_labels(tokens):
    """
    Bagimlilik agacini kullanarak CLAUSE etiketlerini hesaplar.
    RELCL  : acl:relcl ile baglanan cumlecikler
    COMPCL : ccomp/xcomp ile baglanan tamamlayici cumlecikler
    ADVCL  : advcl ile baglanan zarfsal cumlecikler
    """
    n = len(tokens)
    # 1-tabanli indeks -> (form, upos, deprel, head)
    token_map    = {}
    children_map = defaultdict(list)

    for i, (form, upos, deprel, head, idx) in enumerate(tokens):
        token_map[idx] = (form, upos, deprel, head)
        if head > 0:
            children_map[head].append(idx)

    # Her token icin clause tipini bul
    clause_type = {idx: "O" for _, _, _, _, idx in tokens}

    for idx, (form, upos, deprel, head) in token_map.items():
        # Bu token bir cumlecik mi?
        if deprel in RELCL_DEPRELS:
            subtree = get_subtree(idx, children_map)
            for node in subtree:
                if node in clause_type:
                    clause_type[node] = "RELCL"
        elif deprel in COMPCL_DEPRELS:
            subtree = get_subtree(idx, children_map)
            for node in subtree:
                if node in clause_type:
                    clause_type[node] = "COMPCL"
        elif deprel in ADVCL_DEPRELS:
            subtree = get_subtree(idx, children_map)
            for node in subtree:
                if node in clause_type:
                    clause_type[node] = "ADVCL"

    # BIO formata cevir - art arda aynı clause tipi ise B/I
    clause_labels = []
    prev_clause = None
    for _, _, _, _, idx in tokens:
        ct = clause_type[idx]
        if ct == "O":
            clause_labels.append("O")
            prev_clause = None
        elif ct != prev_clause:
            clause_labels.append(f"B-{ct}")
            prev_clause = ct
        else:
            clause_labels.append(f"I-{ct}")

    return clause_labels


# ========================================
# 3. CoNLL-U Dosyasini Oku
# ========================================

def parse_conllu(filepath):
    """
    CoNLL-U dosyasini okur.
    Her cumle icin: [(form, upos, deprel, head, idx), ...] doner.
    """
    sentences = []
    current   = []

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            if line == "":
                if current:
                    sentences.append(current)
                    current = []
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                continue
            # Cok-kelimeli token satirlarini atla
            if "-" in parts[0] or "." in parts[0]:
                continue

            idx    = int(parts[0])
            form   = parts[1]
            upos   = parts[3]
            deprel = parts[7]
            head   = int(parts[6]) if parts[6].isdigit() else 0

            current.append((form, upos, deprel, head, idx))

    if current:
        sentences.append(current)

    return sentences


# ========================================
# 4. 3 Sutunlu CoNLL Formatina Cevir ve Kaydet
# ========================================

def convert_to_full_conll(sentences, output_path):
    """
    3 sutunlu CoNLL formatinda kaydeder:
    ID  FORM  CHUNK-OUTER  CHUNK-INNER  CLAUSE
    (CHUNK-INNER = CLAUSE ile ayni, ic ice gecen yapilari gosterir)
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for tokens in sentences:
            text = " ".join(t[0] for t in tokens)
            f.write(f"# text = {text}\n")
            f.write(f"# columns = ID FORM CHUNK-OUTER CHUNK-INNER CLAUSE\n")

            outer_labels  = bio_chunk_sequence(tokens)
            clause_labels = compute_clause_labels(tokens)

            for i, (token, outer, clause) in enumerate(
                    zip(tokens, outer_labels, clause_labels)):
                form = token[0]
                idx  = token[4]
                # CHUNK-INNER: Clause olmayan kelimeler icin _
                inner = clause if clause != "O" else "_"
                f.write(f"{idx}\t{form}\t{outer}\t{inner}\t{clause}\n")
            f.write("\n")

    print(f"  OK - Kaydedildi: {output_path} ({len(sentences)} cumle)")


# ========================================
# 5. Tum Parcalari Birlestir ve Kaydet
# ========================================

print("\n" + "=" * 60)
print("CoNLL-U -> 3 Sutunlu Chunk CoNLL formatina donusuturuluyor...")
print("=" * 60)

all_sentences = []
for split in ["train", "dev"]:
    path = os.path.join(DATA_DIR, f"ud_turkish_{split}.conllu")
    if os.path.exists(path):
        sents = parse_conllu(path)
        all_sentences.extend(sents)
        print(f"  [{split}] {len(sents)} cumle okundu.")

test_sentences = []
test_path = os.path.join(DATA_DIR, "ud_turkish_test.conllu")
if os.path.exists(test_path):
    test_sentences = parse_conllu(test_path)
    print(f"  [test]  {len(test_sentences)} cumle okundu.")

random.seed(42)
random.shuffle(all_sentences)

if not test_sentences:
    split_idx      = int(len(all_sentences) * 0.8)
    train_sentences = all_sentences[:split_idx]
    test_sentences  = all_sentences[split_idx:]
else:
    train_sentences = all_sentences

print(f"\n  Egitim : {len(train_sentences)} cumle")
print(f"  Test   : {len(test_sentences)} cumle")

convert_to_full_conll(train_sentences, os.path.join(DATA_DIR, "train.conll"))
convert_to_full_conll(test_sentences,  os.path.join(DATA_DIR, "test.conll"))

print("\n[TAMAM] Veri hazirlama tamamlandi!")

# ========================================
# 6. Etiket Dagilimi Goster
# ========================================
from collections import Counter

outer_counts  = Counter()
clause_counts = Counter()

for tokens in train_sentences:
    for label in bio_chunk_sequence(tokens):
        outer_counts[label] += 1
    for label in compute_clause_labels(tokens):
        clause_counts[label] += 1

print("\nCHUNK-OUTER etiket dagilimi:")
print("-" * 35)
for label, count in sorted(outer_counts.items()):
    print(f"  {label:<12} : {count:>6} token")

print("\nCLAUSE etiket dagilimi:")
print("-" * 35)
for label, count in sorted(clause_counts.items()):
    print(f"  {label:<12} : {count:>6} token")
