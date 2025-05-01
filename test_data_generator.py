import os
import shutil
import pandas as pd

# Create project folder structure
project_name = "AI_Predictor_testData"
os.makedirs(project_name, exist_ok=True)

# Create subfolders for inputs and outputs
os.makedirs(f"{project_name}/data", exist_ok=True)
os.makedirs(f"{project_name}/outputs", exist_ok=True)
os.makedirs(f"{project_name}/models", exist_ok=True)

# Prepare sample genotype_matrix.csv
genotype_sample = pd.DataFrame(
    [[1, 0, 1, 1, 0], [0, 1, 1, 0, 1], [1, 1, 0, 0, 0],[1, 0, 1, 1, 0], [0, 1, 1, 0, 1], [1, 1, 0, 0, 0],[1, 0, 1, 1, 0], [0, 1, 1, 0, 1], [1, 1, 0, 0, 0]],
    columns=["geneA", "geneB", "SNP1", "SNP2", "SNP3"],
    index=["Sample1", "Sample2", "Sample3","Sample4","Sample5","Sample6","Sample7","Sample8","Sample9"]
)
genotype_sample.index.name = "Sample"
genotype_sample.to_csv(f"{project_name}/data/genotype_matrix.csv")

# Prepare sample resistance_labels.csv
resistance_sample = pd.DataFrame(
    [[1, 0, 1], [0, 1, 0], [1, 1, 1],[1, 0, 1], [0, 1, 0], [1, 1, 1],[1, 0, 1], [0, 1, 0], [1, 1, 1]],
    columns=["Azithromycin", "Ciprofloxacin", "Mecillinam"],
    index=["Sample1", "Sample2", "Sample3","Sample4","Sample5","Sample6","Sample7","Sample8","Sample9"]
)
resistance_sample.index.name = "Sample"
resistance_sample.to_csv(f"{project_name}/data/resistance_labels.csv")









