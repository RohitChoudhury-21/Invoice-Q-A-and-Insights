import pandas as pd
import numpy as np

def compute_stats(csv_path: str = "data/clean_invoices.csv"):
    df = pd.read_csv(csv_path)
    # Convert date
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df = df.dropna(subset=["invoice_date"])

    total_spend = df["total_amount_usd"].sum()
    count = len(df)
    avg_amt = df["total_amount_usd"].mean()
    median_amt = df["total_amount_usd"].median()
    spend_per_vendor = df.groupby("vendor")["total_amount_usd"].sum().sort_values(ascending=False)
    spend_per_month = df.set_index("invoice_date").resample("ME")["total_amount_usd"].sum()    
    top5_vendors = spend_per_vendor.head(5)
    mom_change = spend_per_month.pct_change() * 100
    mean_amt = df["total_amount_usd"].mean()
    std_amt = df["total_amount_usd"].std()
    outliers = df[df["total_amount_usd"] > mean_amt + 3 * std_amt]

    stats = {
        "total_spend": total_spend,
        "invoice_count": count,
        "average_amount": avg_amt,
        "median_amount": median_amt,
        "spend_per_vendor": spend_per_vendor,
        "spend_per_month": spend_per_month,
        "top5_vendors": top5_vendors,
        "month_over_month_change": mom_change,
        "outliers": outliers[["invoice_number", "vendor", "total_amount_usd"]]
    }
    return stats

if __name__ == "__main__":
    stats = compute_stats()
    print("Total spend:", stats["total_spend"])
    print("Invoice count:", stats["invoice_count"])
    print("Average amount:", stats["average_amount"])
    print("Median amount:", stats["median_amount"])
    print("Top 5 vendors:")
    print(stats["top5_vendors"])
    print("Monthly spend:")
    print(stats["spend_per_month"])
    print("Outliers:")
    print(stats["outliers"])