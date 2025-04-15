import pandas as pd

def save_matrix_to_excel(matrix, cv_filenames, job_filenames, filename):
    """Salvează matricea de scoruri în Excel"""
    df = pd.DataFrame(matrix, index=cv_filenames, columns=job_filenames)
    df.to_excel(filename)
    print(f"Fișierul Excel a fost salvat: {filename}")