import pandas as pd
from pathlib import Path
from src import preprocess
from src.generate_gpt import generate_with_gpt

BASE = Path(__file__).resolve().parent

def main():
    # 1) Load data
    stories, criteria = preprocess.load_data(
        str(BASE / "data/user_stories.xlsx"),
        str(BASE / "data/acceptance_criteria.xlsx"),
    )

    if stories.empty:
        print("Nu exista user stories in data/user_stories.xlsx")
        return

    all_generated = []

    # 2) Pentru fiecare story, adunam AC si generam test cases
    for _, row in stories.iterrows():
        sid = row["story_id"]
        story_text = str(row["user_story"])

        ac_list = criteria[criteria["story_id"] == sid]["ac_text"].dropna().astype(str).tolist()
        if not ac_list:
            ac_list = []

        gen_cases = generate_with_gpt(story_text, ac_list)
        # adauga metadata
        for i, c in enumerate(gen_cases, start=1):
            c.update({
                "story_id": sid,
                "tc_id": f"AIGEN-{sid}-{i}",
                "type": "generated"
            })
        all_generated.extend(gen_cases)

    if not all_generated:
        print("Modelul nu a intors niciun test case. Verifica .env si modelul.")
        return

    # 3) Salvam in results/report.xlsx (foaia 'generated')
    out = pd.DataFrame(all_generated)[
        ["story_id","tc_id","title","preconditions","steps","data","expected","type"]
    ]
    out_path = BASE / "results" / "report.xlsx"

    # Pastreaza celelalte foi, doar actualizeaza 'generated'
    try:
        with pd.ExcelWriter(out_path, mode="a", if_sheet_exists="replace") as w:
            out.to_excel(w, sheet_name="generated", index=False)
    except FileNotFoundError:
        with pd.ExcelWriter(out_path) as w:
            out.to_excel(w, sheet_name="generated", index=False)

    print(f"✔ Testele generate au fost salvate in: {out_path}")

if __name__ == "__main__":
    main()
