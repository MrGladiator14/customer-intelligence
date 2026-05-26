"""Meridian Customer Intelligence Platform — Dummy Data Generator."""

import csv
from src.config import DATA_DIR

# 10-row training data
TRAIN_RECORDS = [
    {
        "customer_id": "CUST001",
        "age": 34,
        "education": "tertiary",
        "job": "management",
        "balance": 1500.50,
        "duration": 240,
        "complaint": "Extremely disappointed with the high transaction fees charged on my savings account without prior notice. [Doc-101]",
        "converted": 1
    },
    {
        "customer_id": "CUST002",
        "age": 45,
        "education": "secondary",
        "job": "technician",
        "balance": 450.00,
        "duration": 90,
        "complaint": "I had a very long delay when trying to transfer money using the mobile app during the weekend. [Doc-102]",
        "converted": 0
    },
    {
        "customer_id": "CUST003",
        "age": 28,
        "education": "tertiary",
        "job": "self-employed",
        "balance": 12000.00,
        "duration": 500,
        "complaint": "The customer support team took over three days to respond to my fraudulent transaction query. Unacceptable! [Doc-103]",
        "converted": 1
    },
    {
        "customer_id": "CUST004",
        "age": 52,
        "education": "primary",
        "job": "blue-collar",
        "balance": -120.50,
        "duration": 15,
        "complaint": "My debit card was blocked without any reason while I was trying to purchase groceries. [Doc-104]",
        "converted": 0
    },
    {
        "customer_id": "CUST005",
        "age": 39,
        "education": "secondary",
        "job": "services",
        "balance": 3500.25,
        "duration": 310,
        "complaint": "The monthly bank statements are showing incorrect balance. The fees are not transparently detailed. [Doc-105]",
        "converted": 1
    },
    {
        "customer_id": "CUST006",
        "age": 61,
        "education": "unknown",
        "job": "retired",
        "balance": 50000.00,
        "duration": 400,
        "complaint": "The branch staff were incredibly helpful when I had issues setting up my retirement plan savings account. [Doc-106]",
        "converted": 1
    },
    {
        "customer_id": "CUST007",
        "age": 42,
        "education": "secondary",
        "job": "admin.",
        "balance": 0.00,
        "duration": 45,
        "complaint": "My online banking account was locked and the password reset link did not work for multiple hours. [Doc-107]",
        "converted": 0
    },
    {
        "customer_id": "CUST008",
        "age": 23,
        "education": "tertiary",
        "job": "student",
        "balance": 120.00,
        "duration": 180,
        "complaint": "I was charged an overdraft fee even though my account balance never dropped below zero. [Doc-108]",
        "converted": 0
    },
    {
        "customer_id": "CUST009",
        "age": 50,
        "education": "tertiary",
        "job": "entrepreneur",
        "balance": 8900.00,
        "duration": 600,
        "complaint": "The loan approval process takes too long. I submitted all documents weeks ago but have no response. [Doc-109]",
        "converted": 1
    },
    {
        "customer_id": "CUST010",
        "age": 31,
        "education": "secondary",
        "job": "management",
        "balance": 2200.75,
        "duration": 120,
        "complaint": "Received multiple marketing calls every day despite having opted out of the telemarketing list. [Doc-110]",
        "converted": 0
    }
]

# 10-row testing data
TEST_RECORDS = [
    {
        "customer_id": "CUST101",
        "age": 35,
        "education": "tertiary",
        "job": "management",
        "balance": 2000.00,
        "duration": 220,
        "complaint": "The interest rate on my home loan was increased suddenly without any prior email notification. [Doc-201]",
        "converted": 1
    },
    {
        "customer_id": "CUST102",
        "age": 41,
        "education": "secondary",
        "job": "technician",
        "balance": 600.00,
        "duration": 110,
        "complaint": "Mobile app keeps crashing whenever I try to scan a QR code for payments. [Doc-202]",
        "converted": 0
    },
    {
        "customer_id": "CUST103",
        "age": 29,
        "education": "tertiary",
        "job": "self-employed",
        "balance": 15000.00,
        "duration": 480,
        "complaint": "Excellent service at the downtown branch. The staff helped me resolve credit card issues in minutes. [Doc-203]",
        "converted": 1
    },
    {
        "customer_id": "CUST104",
        "age": 55,
        "education": "primary",
        "job": "blue-collar",
        "balance": 250.00,
        "duration": 60,
        "complaint": "Unable to withdraw cash from the local ATM as it is out of service constantly during weekends. [Doc-204]",
        "converted": 0
    },
    {
        "customer_id": "CUST105",
        "age": 38,
        "education": "secondary",
        "job": "services",
        "balance": 2800.00,
        "duration": 340,
        "complaint": "Highly dissatisfied with the double billing on my monthly credit card fees. [Doc-205]",
        "converted": 1
    },
    {
        "customer_id": "CUST106",
        "age": 65,
        "education": "tertiary",
        "job": "retired",
        "balance": 75000.00,
        "duration": 410,
        "complaint": "My online stock trading account is locked and customer support has not replied to my emails. [Doc-206]",
        "converted": 1
    },
    {
        "customer_id": "CUST107",
        "age": 44,
        "education": "secondary",
        "job": "admin.",
        "balance": -50.00,
        "duration": 50,
        "complaint": "Credit card payment was declined at a merchant location despite having a positive credit limit. [Doc-207]",
        "converted": 0
    },
    {
        "customer_id": "CUST108",
        "age": 24,
        "education": "secondary",
        "job": "services",
        "balance": 300.00,
        "duration": 150,
        "complaint": "I was charged a foreign transaction fee for an online payment that was billed in USD. [Doc-208]",
        "converted": 0
    },
    {
        "customer_id": "CUST109",
        "age": 48,
        "education": "tertiary",
        "job": "entrepreneur",
        "balance": 9200.00,
        "duration": 550,
        "complaint": "The auto loan interest rate offered is higher than the rate on your website promotional banner. [Doc-209]",
        "converted": 1
    },
    {
        "customer_id": "CUST110",
        "age": 32,
        "education": "secondary",
        "job": "management",
        "balance": 1800.00,
        "duration": 130,
        "complaint": "I was registered for standard SMS notifications without consent and am being charged monthly. [Doc-210]",
        "converted": 0
    }
]

def write_csv(filepath, records):
    keys = records[0].keys()
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    print(f"Successfully wrote {len(records)} records to {filepath}")

def main():
    write_csv(DATA_DIR / "train.csv", TRAIN_RECORDS)
    write_csv(DATA_DIR / "test.csv", TEST_RECORDS)

if __name__ == "__main__":
    main()
