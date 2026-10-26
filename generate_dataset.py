"""
Dataset Generation Script: Realistic Raw Customer Banking & Credit Features.
Injects intentional anomalies:
  1. Exact duplicate rows
  2. Missing values (nulls) in numerical and categorical features
  3. Dirty string fields with whitespace and irregular capitalization
  4. Outliers in numerical distributions (income, credit score, age)
"""

import random
import csv
from pathlib import Path
from config import RAW_DATA_PATH, DATA_DIR

def generate_raw_dataset(num_rows: int = 2500, random_seed: int = 42):
    random.seed(random_seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    first_names = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", 
                   "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", 
                   "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Christopher", 
                   "Nancy", "Daniel", "Lisa", "Matthew", "Betty", "Anthony", "Margaret"]
    
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", 
                  "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", 
                  "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"]

    education_levels = ["High School", "Bachelor", "Master", "PhD", "Associate"]
    employment_statuses = ["Employed", "Self-Employed", "Unemployed", "Retired"]
    genders = ["Male", "Female", "Non-Binary"]
    
    rows = []

    for i in range(1, num_rows + 1):
        cust_id = f"CUST_{1000 + i}"
        
        # Inconsistent name strings (leading/trailing whitespace, mixed cases)
        first = random.choice(first_names)
        last = random.choice(last_names)
        raw_name = f"{first} {last}"
        casing_choice = random.random()
        if casing_choice < 0.2:
            raw_name = raw_name.lower()
        elif casing_choice < 0.4:
            raw_name = raw_name.upper()
        if random.random() < 0.35:
            raw_name = f"  {raw_name}   "

        # Age: with intentional nulls and outliers
        if random.random() < 0.06:
            age = ""
        elif random.random() < 0.02:
            age = random.choice([-3, 142, 180]) # invalid / extreme outliers
        else:
            age = random.randint(21, 68)

        # Gender: with occasional nulls
        gender = "" if random.random() < 0.05 else random.choice(genders)

        # Education: with occasional nulls and dirty casing
        if random.random() < 0.07:
            edu = ""
        else:
            edu = random.choice(education_levels)
            if random.random() < 0.15:
                edu = f" {edu.lower()} "

        # Employment: with occasional nulls
        emp = "" if random.random() < 0.06 else random.choice(employment_statuses)

        # Annual Income: log-normal distribution with missing values and extreme high outliers
        if random.random() < 0.08:
            income = ""
        elif random.random() < 0.02:
            income = round(random.uniform(900000, 2500000), 2) # high outlier
        else:
            base_income = random.gauss(65000, 22000)
            income = round(max(18000.0, base_income), 2)

        # Credit Score: 300 to 850 with missing values and erroneous values
        if random.random() < 0.07:
            credit_score = ""
        elif random.random() < 0.02:
            credit_score = random.choice([150, 999, 1200]) # erroneous outliers
        else:
            credit_score = int(random.gauss(680, 75))
            credit_score = max(300, min(850, credit_score))

        # Total Debt: with missing values
        debt = "" if random.random() < 0.05 else round(random.uniform(1000, 75000), 2)

        # Loan Amount
        loan_amt = round(random.uniform(3000, 50000), 2)

        # Account Balance: with missing values
        balance = "" if random.random() < 0.06 else round(random.uniform(500, 95000), 2)

        # Delinquent Months
        delinq = random.choices([0, 1, 2, 3, 4, 6], weights=[70, 15, 7, 4, 2, 2])[0]

        # Loan Approval Logic (Ground truth with slight noise)
        score_val = credit_score if credit_score != "" and isinstance(credit_score, int) and 300 <= credit_score <= 850 else 600
        inc_val = income if income != "" and isinstance(income, (int, float)) and income < 500000 else 50000
        is_approved = 1 if (score_val >= 660 and inc_val >= 35000 and delinq <= 1) else 0
        if random.random() < 0.05:
            is_approved = 1 - is_approved # label noise

        rows.append([
            cust_id, raw_name, age, gender, edu, emp, 
            income, credit_score, debt, loan_amt, balance, delinq, is_approved
        ])

    # Inject ~120 exact duplicate rows to test PySpark deduplication
    duplicate_rows = random.choices(rows[:500], k=120)
    rows.extend(duplicate_rows)
    random.shuffle(rows)

    fieldnames = [
        "customer_id", "full_name", "age", "gender", "education_level", 
        "employment_status", "annual_income", "credit_score", "total_debt", 
        "loan_amount", "account_balance", "delinquent_months", "loan_approved"
    ]

    with open(RAW_DATA_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        writer.writerows(rows)

    print(f"Generated raw dataset at: {RAW_DATA_PATH}")
    print(f"Total Records Written: {len(rows)} (including {len(duplicate_rows)} injected duplicates)")
    return len(rows)


if __name__ == "__main__":
    generate_raw_dataset()
