import pandas as pd

def save_matrix_to_excel(matrix, cv_filenames, job_filenames, filename='outputs/similarity_results.xlsx'):
    df = pd.DataFrame(matrix, index=cv_filenames, columns=job_filenames)
    df.to_excel(filename)
    print(f"\nExcel file saved in : {filename}")
