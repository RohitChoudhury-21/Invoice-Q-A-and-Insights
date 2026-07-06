import matplotlib.pyplot as plt
import pandas as pd
from app.analytics import compute_stats

def plot_vendor_spend(stats, output="results/vendor_spend.png"):
    top10 = stats["spend_per_vendor"].head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(top10.index, top10.values, color="steelblue")
    ax.set_title("Spend per Vendor (Top 10)")
    ax.set_ylabel("Total Spend (USD)")
    ax.set_xlabel("Vendor")
    ax.set_ylim(0, top10.max() * 1.1)
    # Mark top vendor
    bars[0].set_color("orange")
    ax.annotate("Top Vendor", xy=(0, top10.iloc[0]), xytext=(0.5, top10.max()*0.9),
                arrowprops=dict(arrowstyle="->"))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    print(f"Saved vendor spend chart to {output}")

def plot_monthly_spend(stats, output="results/monthly_spend.png"):
    monthly = stats["spend_per_month"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(monthly.index, monthly.values, marker="o", color="green")
    ax.set_title("Spend per Month")
    ax.set_ylabel("Total Spend (USD)")
    ax.set_xlabel("Month")
    ax.set_ylim(0, monthly.max() * 1.1)
    # Mark peak month
    peak_month = monthly.idxmax()
    peak_value = monthly.max()
    ax.annotate("Peak Month", xy=(peak_month, peak_value),
                xytext=(peak_month, peak_value*0.9),
                arrowprops=dict(arrowstyle="->"))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    print(f"Saved monthly spend chart to {output}")

if __name__ == "__main__":
    stats = compute_stats()
    plot_vendor_spend(stats)
    plot_monthly_spend(stats)