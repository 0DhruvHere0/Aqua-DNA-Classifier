from Bio import SeqIO
from collections import Counter

species_list = []

# First pass: count species
for record in SeqIO.parse("sequence.fasta", "fasta"):

    parts = record.description.split()

    if len(parts) >= 3:

        species = parts[1] + " " + parts[2]

        species_list.append(species)

species_counts = Counter(species_list)

# Keep species with >= 20 sequences
MIN_SEQUENCES = 20

valid_species = {
    species
    for species, count in species_counts.items()
    if count >= MIN_SEQUENCES
}

print("Species kept:", len(valid_species))

# Save filtered sequences
output_file = open("filtered_sequences.fasta", "w")

kept = 0

for record in SeqIO.parse("sequence.fasta", "fasta"):

    parts = record.description.split()

    if len(parts) >= 3:

        species = parts[1] + " " + parts[2]

        if species in valid_species:

            output_file.write(
                f">{record.description}\n{record.seq}\n"
            )

            kept += 1

output_file.close()

print("Sequences kept:", kept)

print("\nFiltered FASTA saved as:")
print("filtered_sequences.fasta")