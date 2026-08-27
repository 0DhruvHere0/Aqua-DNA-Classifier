import itertools
K = 5
CANONICAL = True
MIN_LENGTH = 300
MAX_LENGTH = 1500
CONFIDENCE_THRESHOLD = 50.0
_COMPLEMENT = str.maketrans("ACGT", "TGCA")
def reverse_complement(sequence):
    return sequence.translate(_COMPLEMENT)[::-1]
def build_kmer_vocab(k=K, canonical=CANONICAL):
    if not canonical:
        return [
            ''.join(kmer)
            for kmer in itertools.product("ACGT", repeat=k)
        ]
    vocab = []
    seen = set()
    for kmer_tuple in itertools.product("ACGT", repeat=k):
        kmer = ''.join(kmer_tuple)
        key = min(kmer, reverse_complement(kmer))
        if key not in seen:
            seen.add(key)
            vocab.append(key)
    return vocab
ALL_KMERS = build_kmer_vocab()
KMER_INDEX = {kmer: i for i, kmer in enumerate(ALL_KMERS)}
FEATURE_COLUMNS = ALL_KMERS.copy()
def clean_sequence(sequence):
    return "".join(base for base in sequence.upper() if base in "ACGT")
def extract_species_name(description):
    parts = description.split()
    if len(parts) >= 3:
        return parts[1] + " " + parts[2]
    return None
def kmer_frequencies(sequence, k=K):
    counts = [0] * len(ALL_KMERS)
    total = len(sequence) - k + 1
    if total <= 0:
        return counts
    index = KMER_INDEX
    for i in range(total):
        kmer = sequence[i:i+k]
        idx = index.get(kmer)
        if idx is None and CANONICAL:
            idx = index.get(reverse_complement(kmer))
        if idx is not None:
            counts[idx] += 1
    return [count / total for count in counts]