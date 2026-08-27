from Bio import SeqIO
from collections import Counter
species_list = []
lengths = []
for record in SeqIO.parse("sequence.fasta", "fasta"):
    header = record.description
    sequence = str(record.seq)
    parts = header.split()
    if len(parts) >= 3:
        species = parts[1] + " " + parts[2]
        species_list.append(species)
        lengths.append(len(sequence))
species_counts = Counter(species_list)
print("\n===== DATASET SUMMARY =====\n")
print("Total sequences:")
print(len(species_list))
print("\nUnique species:")
print(len(species_counts))
if len(lengths) > 0:
    average_length = sum(lengths) / len(lengths)
    print("\nAverage sequence length:")
    print(round(average_length, 2))
else:
    print("\nNo valid sequences found")
print("\n===== TOP 10 SPECIES =====\n")
for species, count in species_counts.most_common(10):
    print(f"{species} : {count}")
print("\n===== SAMPLE HEADERS =====\n")
for i, record in enumerate(SeqIO.parse("sequence.fasta", "fasta")):
    print(record.description)
    if i == 4:
        break