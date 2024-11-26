from Bio import SeqIO
import pandas as pd
import numpy as np
from tqdm import tqdm

class DatasetCreator:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = pd.read_csv(file_path, sep="\t")
        self.df = self.df[['curation_status', 'coordinate_hg38', 'seq_hg38']]
        self.df = self.df.dropna()
        self.df['chromosome'] = self.df['coordinate_hg38'].str.split(':').str[0]
        self.df['start'] = self.df['coordinate_hg38'].str.split(':').str[1].str.split('-').str[0].astype(int) - 1
        self.df['end'] = self.df['coordinate_hg38'].str.split(':').str[1].str.split('-').str[1].astype(int) - 1
        self.df['seq_length'] = self.df['seq_hg38'].str.len()
        self.df = self.df[['seq_hg38', 'seq_length', 'chromosome', 'start', 'end', 'curation_status']]
        self.df_positives = self.df[self.df['curation_status'] == 'positive']
        self.df_negatives = self.df[self.df['curation_status'] == 'negative']
        self.df_negatives.reset_index(drop=True, inplace=True)
        self.df_positives = self.df_positives.sort_values(by=['chromosome', 'seq_length']).reset_index(drop=True)
        self.df_negatives = self.df_negatives.sort_values(by=['chromosome', 'seq_length']).reset_index(drop=True)
    
    def get_positives(self) -> pd.DataFrame:
        return self.df_positives
    
    def get_negatives(self) -> pd.DataFrame:
        return self.df_negatives
    
    def get_random_negatives(self, path_to_genome: str) -> pd.DataFrame:
        
        # Get the list of nunique chromosomes and counts
        chromosomes = self.df['chromosome'].unique()
        # chrom_counts = self.df['chromosome'].value_counts()
        
        # Create a DataFrame to store the negative examples
        df_negatives = pd.DataFrame(columns=['seq_hg38', 'seq_length', 'chromosome', 'start', 'end', 'curation_status'])

        # Iterate over the file
        for record in tqdm(SeqIO.parse(path_to_genome, "fasta")):
            if record.id in chromosomes:
                chrom_length = len(record.seq)
                # chrom_count = chrom_counts[record.id]
                # Get the rows from the positive examples
                df_positives = self.df_positives[self.df_positives['chromosome'] == record.id]
                # iterate over the rows
                for i, row in df_positives.iterrows():
                    # Get the start and end coordinates
                    # start = int(row['start'])
                    # end = int(row['end'])
                    # Get the length of the sequence
                    seq_length = int(row['seq_length'])
                    
                    # iterate over the sequence from record to get the subsequence of the same length, avoid overlapping, and no N symbols
                    while True:
                        # Get the random start position
                        random_start = np.random.randint(0, chrom_length - seq_length)
                        # Get the random end position
                        random_end = random_start + seq_length
                        # Check if the random start and end positions are not overlapping with the positive examples
                        if not any((start <= random_start <= end) or (start <= random_end <= end) for start, end in zip(df_positives['start'], df_positives['end'])):
                            # Get the subsequence
                            subsequence = record.seq[random_start:random_end]
                            # Merge tuple to string
                            subsequence = ''.join(subsequence)
                            # Check if there are no N symbols
                            if 'N' not in subsequence:
                                # Create a new row
                                new_row = {'seq_hg38': subsequence, 'seq_length': seq_length, 'chromosome': record.id, 'start': random_start, 'end': random_end, 'curation_status': 'negative'}
                                # Append the new row to the DataFrame
                                df_negatives = pd.concat([df_negatives, pd.DataFrame([new_row])], ignore_index=True)
                                break
        return df_negatives.sort_values(by=['chromosome', 'seq_length']).reset_index(drop=True)