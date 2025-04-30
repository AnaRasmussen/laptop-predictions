#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ----------------------------------------
# LOAD & CLEAN DATA
# ----------------------------------------

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath, encoding='latin1')

    # Clean Ram and Weight columns
    df['Ram'] = df['Ram'].str.replace('GB', '', regex=False).astype(float)
    df['Weight'] = df['Weight'].str.replace('kg', '', regex=False).astype(float)

    # Drop rows with missing values
    df = df.dropna()

    return df

# ----------------------------------------
# VISUALIZATION
# ----------------------------------------

def visualize_selected_columns(df, label_name="Price_euros"):
    numeric_cols = ['Inches', 'Ram', 'Weight', label_name]
    cat_cols = ['Company', 'TypeName', 'OpSys']

    # 1. Histograms for numeric features
    for col in numeric_cols:
        plt.figure(figsize=(8, 5))
        plt.hist(df[col], bins=20, color='skyblue', edgecolor='black')
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"{col.lower()}_histogram.png")
        plt.close()

    # 2. Correlation heatmap
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.close()

    # 3. Average Price by Categorical Columns
    for col in cat_cols:
        top_vals = df[col].value_counts().nlargest(10).index
        filtered = df[df[col].isin(top_vals)]
        avg_price = filtered.groupby(col)[label_name].mean().sort_values(ascending=False)

        plt.figure(figsize=(10, 5))
        sns.barplot(x=avg_price.index, y=avg_price.values, palette="viridis")
        plt.title(f"Average Laptop Price by {col}")
        plt.ylabel("Average Price (€)")
        plt.xlabel(col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"avg_price_by_{col.lower()}.png")
        plt.close()

def generate_text_report(df, label_name="Price_euros", out_filename="laptop_data_report.txt"):
    with open(out_filename, 'w') as f:
        f.write("LAPTOP DATA EXPLORATION REPORT\n")
        f.write("="*40 + "\n\n")

        # Descriptive stats
        f.write("NUMERIC SUMMARY STATISTICS:\n")
        f.write(df[['Inches', 'Ram', 'Weight', label_name]].describe().to_string())
        f.write("\n\n")

        # Top values in categorical columns
        categorical = ['Company', 'TypeName', 'OpSys']
        for col in categorical:
            f.write(f"TOP 10 VALUES IN {col.upper()}:\n")
            f.write(df[col].value_counts().head(10).to_string())
            f.write("\n\n")

        # Avg price by category
        for col in categorical:
            f.write(f"💰 AVERAGE PRICE BY {col.upper()}:\n")
            avg = df.groupby(col)[label_name].mean().sort_values(ascending=False).head(10)
            f.write(avg.to_string())
            f.write("\n\n")

        f.write("Report generated successfully.\n")


# ----------------------------------------
# MAIN
# ----------------------------------------

def main():
    filepath = "laptop-data.csv"
    df = load_and_clean_data(filepath)
    visualize_selected_columns(df, label_name="Price_euros")
    generate_text_report(df, label_name="Price_euros")
    print("Graphs and report saved!")


if __name__ == "__main__":
    main()
