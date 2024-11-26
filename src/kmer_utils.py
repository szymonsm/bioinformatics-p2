from Bio.Seq import reverse_complement
from itertools import product
import pandas as pd

def count_kmers(sequence, k):
    """
    Counts k-mer frequencies in a DNA sequence, including reverse complements.
    
    Args:
        sequence (str): The DNA sequence (uppercase/lowercase treated equally).
        k (int): The k-mer length.
    
    Returns:
        dict: A dictionary with k-mers and their normalized frequencies.
    """
    # Normalize sequence to uppercase
    sequence = sequence.upper()
    
    # Generate canonical k-mers (min of k-mer and its reverse complement)
    kmers = {}
    for p in product('ACGT', repeat=k):
        kmer = ''.join(p)
        rev_comp = reverse_complement(kmer)
        canonical_kmer = min(kmer, rev_comp)
        if canonical_kmer not in kmers:
            kmers[canonical_kmer] = 0
    
    # Slide through the sequence and count k-mers
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        rev_comp = reverse_complement(kmer)
        canonical_kmer = min(kmer, rev_comp)
        if canonical_kmer in kmers:
            kmers[canonical_kmer] += 1
    
    # Normalize frequencies
    sequence_length = len(sequence)
    kmers = {k: v / sequence_length for k, v in kmers.items()}
    
    return kmers

def process_kmer_for_df(df, k):
    """
    Processes a DataFrame of sequences to compute k-mer frequencies for a range of k.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing sequences and metadata.
        k (int): k-mer length.
    
    Returns:
        pd.DataFrame: DataFrame with k-mer features added.
    """
    
    df_kmer = df.copy()
    df_kmer['seq_hg38'] = df_kmer['seq_hg38'].str.upper()

    # Compute k-mer frequencies and add this dict as new columns
    kmers = df_kmer['seq_hg38'].apply(lambda x: count_kmers(x, k))
    kmers = pd.DataFrame(kmers.tolist())
    kmers.columns = [f'{col}' for col in kmers.columns]
    df_kmer = pd.concat([kmers, df_kmer['curation_status']], axis=1)

    # Shuffle the DataFrame
    df_kmer = df_kmer.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df_kmer