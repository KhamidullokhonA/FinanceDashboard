# 💰 Personal Expense Analytics Dashboard

A beginner-friendly personal finance dashboard built with Python and Streamlit.

The app allows users to upload a bank transaction CSV file, automatically process transactions, categorize expenses, and visualize spending patterns.

## Features

- Upload bank transaction CSV files
- Clean and process transaction data using Pandas
- Separate debit and credit transactions
- Create custom expense categories
- Manually change transaction categories
- Save category rules using JSON
- Automatically categorize previously recognized transactions
- Display expense summaries by category
- Visualize spending using interactive Plotly charts
- Display total incoming payments

## Technologies Used

- Python
- Streamlit
- Pandas
- Plotly
- JSON

## How It Works

1. Upload a bank statement in CSV format.
2. The app processes the transaction data.
3. Debit transactions are displayed as expenses.
4. Credit transactions are displayed as payments.
5. Users can create custom categories such as:
   - Food
   - Transportation
   - Shopping
   - Entertainment
6. Users can assign transactions to categories.
7. The application stores transaction descriptions inside `categories.json`.
8. When the same transaction description appears again, the app automatically assigns the saved category.
9. Expense totals are grouped by category and displayed using a pie chart.

## Example CSV Format

The CSV file should contain columns similar to:

| Date | Details | Amount | Debit/Credit |
|------|---------|--------|--------------|
| 01 Sep 2026 | Starbucks | 6.50 | Debit |
| 02 Sep 2026 | Walmart | 42.30 | Debit |
| 03 Sep 2026 | Payroll | 1200.00 | Credit |

The date format currently expected by the application is:

```text
DD MMM YYYY
