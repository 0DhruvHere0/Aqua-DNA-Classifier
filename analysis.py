from Bio import SeqIO
from collections import Counter

# Store species names
species_list = []

# Store sequence lengths
lengths = []

# Read FASTA file
for record in SeqIO.parse("sequence.fasta", "fasta"):

    # Full FASTA header
    header = record.description

    # DNA sequence
    sequence = str(record.seq)

    # Split header into words
    parts = header.split()

    # Make sure header contains enough information
    if len(parts) >= 3:

        # Species name
        species = parts[1] + " " + parts[2]

        # Save species
        species_list.append(species)

        # Save sequence length
        lengths.append(len(sequence))

# Count species frequencies
species_counts = Counter(species_list)

# Print dataset summary
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

# Print top 10 species
print("\n===== TOP 10 SPECIES =====\n")

for species, count in species_counts.most_common(10):

    print(f"{species} : {count}")

# Print first 5 headers
print("\n===== SAMPLE HEADERS =====\n")

for i, record in enumerate(SeqIO.parse("sequence.fasta", "fasta")):

    print(record.description)

    if i == 4:
        break