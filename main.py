import os
import nltk
from preprocessing import load_docx_from_folder, extract_skills
from similarity import compute_final_scores
from export_results import save_matrix_to_excel

nltk.download('stopwords')

# Configurații
cv_folder = 'DataSet/cv'
job_folder = 'DataSet/job_descriptions'
output_file = 'outputs/final_scores.xlsx'

# Lista de skill-uri predefinite (trebuie să fie aceeași ca în preprocessing.py)
PREDEFINED_SKILLS = [
    "python", "java", "react", "sql", "javascript",
    "html", "css", "c++", "machine learning", "flask"
]

def get_technical_weights():
    """Obține skill-urile și ponderile de la utilizator"""
    print("Introdu skill-urile tehnice și ponderile (suma=100%):")
    print(f"Skill-uri disponibile: {', '.join(PREDEFINED_SKILLS)}")
    
    skills = {}
    total = 0
    while total < 100:
        skill = input("\nSkill (sau 'stop' pentru a termina): ").strip().lower()
        
        if skill == 'stop':
            break
            
        if skill not in PREDEFINED_SKILLS:
            print(f"⚠ Skill '{skill}' nu este în lista predefinită. Încearcă altul.")
            continue
            
        try:
            weight = int(input(f"Pondere pentru '{skill}' (%): "))
            if weight <= 0:
                print("Ponderea trebuie să fie pozitivă.")
                continue
                
            skills[skill] = weight
            total += weight
            
            if total > 100:
                print(f"⚠ Suma depășește 100% (current: {total}%). Resetez ultima intrare.")
                total -= weight
                del skills[skill]
                
        except ValueError:
            print("⚠ Te rog introdu un număr valid pentru pondere.")
    
    if total != 100:
        print(f"\n⚠ Atenție: Suma ponderilor este {total}% (trebuie exact 100%).")
        print("Voi normaliza ponderile la 100% automat.")
        # Normalizează ponderile
        for skill in skills:
            skills[skill] = round(skills[skill] * 100 / total)
    
    return skills

def main():
    # Încărcare date
    print("\nSe încarcă CV-urile și job-urile...")
    cv_texts, cv_files = load_docx_from_folder(cv_folder)
    job_texts, job_files = load_docx_from_folder(job_folder)
    
    # Verifică datele încărcate
    if not cv_texts or not job_texts:
        print("⛔ Eroare: Nu s-au găsit CV-uri sau job-uri în folderele specificate.")
        return
        
    print(f"✅ S-au încărcat {len(cv_texts)} CV-uri și {len(job_texts)} job-uri.")
    
    # Obține ponderile
    technical_weights = get_technical_weights()
    print("\nPonderi finale:", technical_weights)
    
    # Calcul scoruri
    print("\nSe calculează potrivirile...")
    try:
        final_scores = compute_final_scores(cv_texts, job_texts, technical_weights)
    except Exception as e:
        print(f"⛔ Eroare la calculul potrivirilor: {str(e)}")
        return
    
    # Afișare rezultate
    print("\n🔍 Rezultate:")
    for i, cv_file in enumerate(cv_files):
        best_match_idx = max(range(len(job_files)), key=lambda j: final_scores[i][j])
        print(f"\n📄 CV: {cv_file}")
        print(f"   🏆 Best Match: {job_files[best_match_idx]} (scor: {final_scores[i][best_match_idx]:.2f})")
        for j, job_file in enumerate(job_files):
            print(f"   • {job_file}: {final_scores[i][j]:.2f}")
    
    # Salvare
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_matrix_to_excel(final_scores, cv_files, job_files, output_file)
    print(f"\n💾 Rezultatele au fost salvate în {output_file}")

if __name__ == "__main__":
    main()