from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
DATA.mkdir(parents=True, exist_ok=True)

def main():
    # requirements.xlsx
    req = pd.DataFrame([
        {"requirement_id":"REQ-LOGIN", "requirements_name":"Login", "requirements_description":"User can authenticate with email/password",
         "requirements_rationale":"Secure access", "requirements_platform":"Web", "requirement_details":"Consider lockout after 5 attempts"},
        {"requirement_id":"REQ-SIGNUP", "requirements_name":"Sign up", "requirements_description":"New user registration form",
         "requirements_rationale":"Acquire users", "requirements_platform":"Web", "requirement_details":"Password policy: 8+ chars"},
        {"requirement_id":"REQ-SEARCH", "requirements_name":"Search", "requirements_description":"Keyword search across products",
         "requirements_rationale":"Findability", "requirements_platform":"Web", "requirement_details":"Rate-limit 10 req/s"},
    ])
    req.to_excel(DATA / "requirements.xlsx", index=False)

    # acceptance_criteria.xlsx
    ac = pd.DataFrame([
        {"acceptance_criteria_story_id":"REQ-LOGIN","acceptance_criteria":"Valid credentials redirect to dashboard","acceptance_criteria_details":"Case sensitivity"},
        {"acceptance_criteria_story_id":"REQ-LOGIN","acceptance_criteria":"Invalid password shows error","acceptance_criteria_details":"No timing leak"},
        {"acceptance_criteria_story_id":"REQ-SIGNUP","acceptance_criteria":"Weak password rejected","acceptance_criteria_details":"Feedback message shown"},
        {"acceptance_criteria_story_id":"REQ-SEARCH","acceptance_criteria":"Empty query returns helper text","acceptance_criteria_details":"No results call"},
        {"acceptance_criteria_story_id":"REQ-SEARCH","acceptance_criteria":"Special chars handled safely","acceptance_criteria_details":"No XSS"},
    ])
    ac.to_excel(DATA / "acceptance_criteria.xlsx", index=False)

    # use_cases.xlsx (fields that preprocess combină în uc_text)
    uc = pd.DataFrame([
        {"use_cases_story_id":"REQ-LOGIN","use_cases_title":"Successful login","use_cases_main_flow":"Open /login -> enter valid creds -> submit -> dashboard"},
        {"use_cases_story_id":"REQ-SIGNUP","use_cases_title":"Happy signup","use_cases_main_flow":"Go /signup -> fill fields -> accept T&C -> submit -> verify email"},
        {"use_cases_story_id":"REQ-SEARCH","use_cases_title":"Basic search","use_cases_main_flow":"Type keyword -> press Enter -> result list appears"},
    ])
    uc.to_excel(DATA / "use_cases.xlsx", index=False)

    # manual_cases.xlsx (optional baseline)
    man = pd.DataFrame([
        {"Requirement ID":"REQ-LOGIN","Requirement Name":"Login","Test Case ID":"MAN-1","Title":"Login with valid credentials",
         "Preconditions":"User registered","Steps":"1. Open /login\n2. Enter valid email/pass\n3. Submit","Test Data":"email, pass",
         "Expected Result":"Dashboard is shown","Category":"Positive","Gherkin":"Given registered user...","Source":"Manual (baseline)"},
        {"Requirement ID":"REQ-SIGNUP","Requirement Name":"Sign up","Test Case ID":"MAN-2","Title":"Reject weak password",
         "Preconditions":"N/A","Steps":"1. Open /signup\n2. Enter weak pass\n3. Submit","Test Data":"weak pass",
         "Expected Result":"Validation error displayed","Category":"Negative","Gherkin":"","Source":"Manual (baseline)"},
    ])
    man.to_excel(DATA / "manual_cases.xlsx", index=False)

    print("Demo data written to /data")

if __name__ == "__main__":
    main()
