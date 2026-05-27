import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

# ── Configuration ──────────────────────────────────────────────────────────────
TOTAL_ROWS   = 5000
MIN_USERS    = 700
TARGET_USERS = 750          # a comfortable margin above the minimum

# ── Realistic complaint / query templates ─────────────────────────────────────
COMPLAINTS = [
    # Fees & charges
    "I was charged an unexpected maintenance fee on my savings account without any prior notification.",
    "My account was debited twice for the same transaction. I need an immediate refund.",
    "The ATM withdrawal fee is excessively high compared to other banks. Please review your fee structure.",
    "I noticed an unknown annual fee deducted from my account. This was not mentioned when I opened the account.",
    "Hidden charges appeared on my credit card statement that were never disclosed at the time of application.",
    "I was charged a late payment fee even though I paid before the due date. Please investigate.",
    "An international transaction fee was applied to a domestic purchase. This is clearly a system error.",
    "My account was charged a minimum balance penalty even though my balance was above the required threshold.",
    # Card issues
    "My debit card was declined at a supermarket despite having sufficient balance in my account.",
    "My credit card was blocked without any warning while I was travelling abroad.",
    "I lost my debit card and reported it, but a replacement card has not arrived after three weeks.",
    "The contactless payment feature on my new debit card is not working at any terminal.",
    "My card PIN was changed without my consent. I suspect unauthorized access to my account.",
    "I received a new card but it was not activated despite following all the steps correctly.",
    "My card was cloned and several fraudulent transactions appeared on my statement.",
    # Online & mobile banking
    "The mobile banking app crashes every time I try to initiate a fund transfer.",
    "I have been unable to log in to internet banking for the past two days due to a technical error.",
    "The OTP for online transactions is not being delivered to my registered mobile number.",
    "My online banking password reset link expired within minutes, making it impossible to regain access.",
    "The mobile app shows incorrect balance figures that do not match my passbook.",
    "Scheduled recurring payments through the app were not processed on the specified dates.",
    "Two-factor authentication is not working on my Android device after the latest app update.",
    "I cannot view my transaction history beyond the last 30 days in the mobile app.",
    # Loans & credit
    "My home loan EMI was debited twice this month. I need the extra amount reversed immediately.",
    "I submitted all loan documents three weeks ago but have not received any update on my application.",
    "The interest rate on my personal loan was increased without any written communication.",
    "My credit score dropped significantly after a loan was incorrectly marked as delinquent in your records.",
    "I pre-paid part of my loan but the outstanding balance in the app has not been updated.",
    "The loan repayment schedule provided at disbursement does not match the amounts being debited.",
    "I was promised a lower interest rate but my loan agreement shows a higher rate than agreed.",
    # Transfers & payments
    "A fund transfer I initiated three days ago has not reached the beneficiary account.",
    "I transferred money to the wrong account by mistake. Please help me recover the funds.",
    "My NEFT payment was rejected but the amount was still debited from my account.",
    "I was not notified when my standing instruction payment failed, causing a delayed bill payment.",
    "The transaction shows successful in the app but the recipient says they have not received the money.",
    "An international wire transfer I placed last week is still in pending status.",
    # Customer service
    "I waited on hold for over 45 minutes and my issue was still not resolved by your support team.",
    "I visited the branch three times to update my KYC, but it is still showing as incomplete in the system.",
    "Your customer care executive was rude and dismissive when I called to report a fraud.",
    "I have sent four emails regarding my issue and none of them have received a response.",
    "The branch manager was unhelpful and refused to escalate my complaint to the regional office.",
    "My complaint raised 10 days ago has still not been assigned a reference number.",
    "I was given incorrect information by a phone banking agent, which led to financial loss.",
    # Account management
    "My account was closed without prior notice even though there were no violations on my part.",
    "My registered mobile number and email address were changed without my authorisation.",
    "I have been trying to update my address for two months but the request keeps getting rejected.",
    "The nominee details I submitted are not reflecting correctly in my account profile.",
    "My joint account holder passed away and I need assistance converting it to a single account.",
    "My account has been frozen due to a KYC mismatch that I have already clarified with the branch.",
    "I am unable to upgrade my savings account to a premium account despite meeting all eligibility criteria.",
    # Statements & passbook
    "My monthly account statement shows transactions I did not authorize.",
    "The e-statement email I should receive every month has not arrived for the last three months.",
    "I requested a physical passbook update at the branch but was told the system was down for a week.",
    "There is a discrepancy between my online transaction history and the physical statement I received.",
    # Deposits & withdrawals
    "I deposited cash at the branch counter but the amount is not reflecting in my account after 48 hours.",
    "The ATM swallowed my card without dispensing cash and the transaction was marked as successful.",
    "I tried to withdraw cash from the ATM but it ran out of notes midway and did not credit the amount back.",
    "A fixed deposit that matured last week has not been credited to my account.",
    # Insurance & investments
    "The insurance premium linked to my account was deducted twice in the same month.",
    "I cancelled my mutual fund investment two weeks ago but the redemption amount has not been credited.",
    "My recurring deposit matured but the interest credited is lower than the rate promised at opening.",
    # Privacy & security
    "I received a phishing SMS appearing to come from your bank. Please investigate the source.",
    "My personal details were shared with a third party without my consent.",
    "I noticed an unfamiliar device logged into my net banking account. Please lock my account immediately.",
]

QUERIES = [
    # Account queries
    "Can you explain the difference between a current account and a savings account?",
    "What is the minimum balance requirement for a premium savings account?",
    "How many free ATM transactions am I entitled to per month?",
    "What documents are required to open a joint account?",
    "Can I have more than one savings account at your bank?",
    "What is the process for converting my existing account to a zero-balance account?",
    "How do I add a nominee to my existing bank account?",
    "What happens to my account if I do not maintain the minimum balance?",
    # Card queries
    "How do I apply for a new credit card online?",
    "What is the credit limit on a standard Visa credit card?",
    "How can I check my credit card reward points balance?",
    "What is the process for disputing a transaction on my credit card?",
    "Can I increase my debit card daily withdrawal limit?",
    "How long does it take to receive a replacement debit card after reporting it lost?",
    "Are there any charges for using my card at international ATMs?",
    "Can I set spending limits on my credit card for specific categories?",
    # Loan queries
    "What is the current interest rate for a home loan?",
    "What is the maximum tenure for a personal loan?",
    "How do I check the status of my loan application?",
    "Can I make a partial prepayment on my home loan without penalty?",
    "What documents are needed to apply for a car loan?",
    "How is the EMI calculated for a personal loan of Rs. 5 lakhs?",
    "What is the processing fee for a home loan?",
    "Can I apply for a loan top-up on my existing home loan?",
    # Digital banking queries
    "How do I register for internet banking?",
    "How do I set up UPI on the mobile banking app?",
    "Is there a transaction limit for NEFT transfers?",
    "How can I enable international transactions on my debit card?",
    "How do I change my internet banking password?",
    "Can I schedule future-dated payments through the mobile app?",
    "What should I do if my mobile banking app is not showing updated balance?",
    "How do I link my account to third-party payment apps?",
    # Fees & interest queries
    "What is the annual fee for a platinum credit card?",
    "Is the processing fee for a personal loan refundable if my application is rejected?",
    "What is the penal interest rate for late credit card payments?",
    "Do you charge a fee for NEFT transfers done through the branch?",
    "What is the interest rate for a savings account?",
    "Are there any charges for requesting a physical account statement?",
    # General & product queries
    "Does your bank offer recurring deposit schemes for senior citizens?",
    "What is the maximum amount I can transfer via IMPS in a single transaction?",
    "Do you offer forex cards for international travel?",
    "How can I apply for a locker facility at the branch?",
    "Does the bank offer health insurance bundled with a savings account?",
    "What are the eligibility criteria for a student loan?",
    "Can I convert my credit card bill into EMI after the purchase?",
    "How do I redeem credit card reward points for cashback?",
    "What is the process to close my fixed deposit before maturity?",
    "Do you offer priority banking services for high-net-worth individuals?",
]

# ── Reference data ─────────────────────────────────────────────────────────────
EDUCATION_LEVELS = ["primary", "secondary", "tertiary", "unknown"]
EDUCATION_WEIGHTS = [0.15, 0.40, 0.40, 0.05]

JOB_TYPES = [
    "management", "technician", "self-employed", "blue-collar",
    "services", "retired", "admin.", "student", "entrepreneur",
    "housemaid", "unknown",
]
JOB_WEIGHTS = [0.14, 0.13, 0.10, 0.12, 0.09, 0.08, 0.10, 0.07, 0.07, 0.05, 0.05]

# ── Generate users ─────────────────────────────────────────────────────────────
def generate_users(n: int) -> pd.DataFrame:
    user_ids = [f"CUST{str(i).zfill(4)}" for i in range(1, n + 1)]
    ages = np.clip(np.round(np.random.normal(40, 12, n)).astype(int), 18, 75)
    educations = np.random.choice(EDUCATION_LEVELS, size=n, p=EDUCATION_WEIGHTS)
    jobs = np.random.choice(JOB_TYPES, size=n, p=JOB_WEIGHTS)

    # Balance: mix of distributions to mimic realistic spread
    base_balance = np.where(
        np.random.rand(n) < 0.05,
        np.random.uniform(-5000, -10, n),           # ~5 % in overdraft
        np.where(
            np.random.rand(n) < 0.50,
            np.random.exponential(3000, n),          # mass of small balances
            np.random.exponential(20000, n),         # tail of larger balances
        )
    )
    balances = np.round(base_balance, 2)

    return pd.DataFrame({
        "customer_id": user_ids,
        "age":         ages,
        "education":   educations,
        "job":         jobs,
        "balance":     balances,
    })

# ── Assign rows to users ───────────────────────────────────────────────────────
def assign_rows_to_users(total_rows: int, n_users: int) -> list[str]:
    """
    Distribute TOTAL_ROWS across n_users with a realistic heavy-tail:
    most users have 3–8 records, a few power-users have many more.
    """
    # Draw per-user row counts from a Poisson; clip to at least 1
    lam = total_rows / n_users
    counts = np.random.poisson(lam, n_users)
    counts = np.maximum(counts, 1)

    # Scale so we hit exactly TOTAL_ROWS
    while counts.sum() < total_rows:
        counts[np.random.randint(n_users)] += 1
    while counts.sum() > total_rows:
        idx = np.random.randint(n_users)
        if counts[idx] > 1:
            counts[idx] -= 1

    user_ids = [f"CUST{str(i).zfill(4)}" for i in range(1, n_users + 1)]
    id_col = []
    for uid, cnt in zip(user_ids, counts):
        id_col.extend([uid] * int(cnt))
    random.shuffle(id_col)
    return id_col

# ── Build the synthetic text column ───────────────────────────────────────────
def pick_text_and_response() -> tuple[str, str]:
    """Pick a complaint or query, removing any embedded Doc-ID artefacts, and generate a response."""
    pool   = COMPLAINTS + QUERIES
    text   = random.choice(pool)
    # Strip legacy [Doc-NNN] style annotations if present
    import re
    text = re.sub(r"\s*\[Doc-\d+\]", "", text).strip()
    
    # Generate a mock response based on keywords
    lower_text = text.lower()
    if "fee" in lower_text or "charge" in lower_text:
        response = "Based on our fee schedule (Doc-102), unexpected charges may relate to minimum balance rules or international transaction fees. We have raised a ticket to review your specific account charges and process any eligible refund."
    elif "card" in lower_text or "pin" in lower_text or "atm" in lower_text:
        response = "For card security (Doc-204), please block your card immediately via the mobile app if lost or compromised. A replacement card typically takes 7-10 business days. We are checking the specific status of your card."
    elif "app" in lower_text or "online" in lower_text or "password" in lower_text or "login" in lower_text:
        response = "We apologize for the technical difficulties (Doc-305). Please try clearing your app cache or resetting your password using the 'Forgot Password' link. If the issue persists, our technical team has been notified."
    elif "loan" in lower_text or "emi" in lower_text:
        response = "Your loan query has been escalated (Doc-401). Interest rate changes or EMI schedules are governed by the terms in your loan agreement. A representative will contact you within 24 hours with specifics."
    elif "transfer" in lower_text or "neft" in lower_text or "payment" in lower_text:
        response = "Fund transfers can sometimes be delayed due to network issues (Doc-508). We are tracking the transaction reference and will credit the funds back if the transfer failed."
    elif "account" in lower_text or "kyc" in lower_text:
        response = "Account or KYC updates require verification (Doc-602). Please ensure you have submitted the latest valid ID proof. We are reviewing your account status and will update you shortly."
    else:
        response = "Thank you for reaching out. Based on our policy documents (Doc-999), we are reviewing your query and a customer support agent will follow up with you shortly with a detailed resolution."
        
    return text, response

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    n_users = TARGET_USERS
    users_df = generate_users(n_users)
    user_lookup = users_df.set_index("customer_id")

    id_col = assign_rows_to_users(TOTAL_ROWS, n_users)

    rows = []
    for customer_id in id_col:
        info = user_lookup.loc[customer_id]
        # Duration: skewed towards shorter calls, some long ones
        duration = int(np.clip(np.random.exponential(200), 10, 900))
        # Converted: roughly proportional to duration + education + balance
        prob_convert = 0.30
        if duration > 300:
            prob_convert += 0.15
        if info["education"] == "tertiary":
            prob_convert += 0.10
        if info["balance"] > 5000:
            prob_convert += 0.10
        converted = int(random.random() < min(prob_convert, 0.85))

        complaint, response = pick_text_and_response()

        rows.append({
            "customer_id": customer_id,
            "age":         int(info["age"]),
            "education":   info["education"],
            "job":         info["job"],
            "balance":     float(info["balance"]),
            "duration":    duration,
            "complaint":   complaint,
            "support_response": response,
            "converted":   converted,
        })

    df = pd.DataFrame(rows)
    # Index IS the unique doc identifier (0-based), so no extra column needed
    df.index.name = "doc_id"

    # ── Sanity checks ──────────────────────────────────────────────────────────
    assert len(df) == TOTAL_ROWS,             f"Row count mismatch: {len(df)}"
    assert df["customer_id"].nunique() >= MIN_USERS, \
        f"Too few unique users: {df['customer_id'].nunique()}"
    assert not df["complaint"].str.contains(r"\[Doc-\d+\]", regex=True).any(), \
        "Doc-ID artefacts found in complaint column"

    out_path = "data/synthetic_train.csv"
    df.to_csv(out_path)
    print(f"[OK] Saved {len(df):,} rows to {out_path}")
    print(f"    Unique users  : {df['customer_id'].nunique()}")
    print(f"    Converted=1   : {df['converted'].sum()} ({df['converted'].mean():.1%})")
    print(f"    Avg duration  : {df['duration'].mean():.0f}s")
    print(f"    Balance range : {df['balance'].min():.2f} – {df['balance'].max():.2f}")
    print()
    print(df.head(10).to_string())

if __name__ == "__main__":
    main()
