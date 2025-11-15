The original dataset was download at [Duplicated antibiotic resistance genes reveal ongoing selection and horizontal gene transfer in bacteria](https://www.nature.com/articles/s41467-024-45638-9)


Files:
Embedding with GenomeOcean-4B-bgcFM.py and Embedding with GenomeOcean-4BM.py are example scripts that we used GenomeOcean models embedding our sequences. (Notices: in this code, we only select the 3000 sequences as an examples, when you use it, please make sure you have revise it accordingly.)


“positive_pairs.csv” this is the final positive datasets after cleaning, both chromosomes and plasmids are noted as their Accession Number in NCBI RefSeq Database
"Supplementary_2.csv" is the original dataset we download at [Maddamsetti, R., Shyti, I., Wilson, M. L., Son, H. I., Baig, Y., Zhou, Z., ... & You, L. (2025). Scaling laws of bacterial and archaeal plasmids. Nature Communications, 16(1), 6023.](https://www.nature.com/articles/s41467-025-61205-2).




📌 File Descriptions

1. Embedding Scripts

Embedding with GenomeOcean-4B.py

Embedding with GenomeOcean-4B-bgcFM.py

Example scripts for generating CDS sequence embeddings using GenomeOcean models.

Note: These example scripts process only ~3,000 sequences.
When using them for full datasets, update the file paths, batch size, and sequence lists accordingly.

⸻

2. Dataset Construction

positive_pairs.csv

Final cleaned positive plasmid–host interaction dataset.
Plasmids and chromosomes are labeled using their NCBI RefSeq accession numbers.

Supplementary_2.csv

Original plasmid dataset downloaded from:

Maddamsetti, R., Shyti, I., Wilson, M. L., Son, H. I., Baig, Y., Zhou, Z., … & You, L. (2025).
Scaling laws of bacterial and archaeal plasmids.
Nature Communications, 16(1), 6023.

Negative pair generation and dataset construction.ipynb

Notebook for generating negative host–plasmid pairs and constructing training/validation/test datasets.

⸻

3. Model Training and Evaluation

Model_trainining_and_evaluation.py

Main PlasmidHostCLIP model training script, including attention-based fusion of CDS embeddings and evaluation metrics.

run_distributed_clip_training_slurm.sh

SLURM batch script for distributed training on HPC clusters.

⸻

4. Data Mining and Preprocessing

web_mining_uniprot.ipynb

Notebook for acquiring CDS sequences from UniProt and performing preprocessing steps.
