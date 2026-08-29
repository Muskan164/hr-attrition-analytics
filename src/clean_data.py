import pandas as pd

raw_path = "data/HR-Employee-Attrition.csv"
clean_path = "data/hr_clean.csv"

# Replacement cost multiplier by job level (rough industry rule of thumb:
# junior roles cost less to replace than senior/specialist roles)
replacement_cost_multiplier = {
    1: 0.5, #entry level
    2:0.75, #associate
    3:1.0, #senior
    4:1.5, #specialist
    5:2.0, #executive
}

def load_and_clean():
    df = pd.read_csv(raw_path)
    # These columns are identical for every row in this dataset - no signal
    constant_cols = ["EmployeeCount", "Over18", "StandardHours"]
    df = df.drop(columns=[col for col in constant_cols if col in df.columns])

    #Encoding the target variable as a binary column for modeling purposes, Keep yes or no for readable charts
    df["AttritionFlag"] = (df["Attrition"] == "Yes").astype(int)

    #Estimate aannual salary from monthly salary, then apply seniority multiplier 
    df["EstAnnualSalary"] = df["MonthlyIncome"] * 12
    df["ReplacementMultiplier"] = df["JobLevel"].map(replacement_cost_multiplier) 
    df["EstReplacementCost"] = df["EstAnnualSalary"] * df["ReplacementMultiplier"]
    #Quick sanity checks - printed so we can see them when we run this file directly
    print("Shape:", df.shape)
    print("Nulls per column:\n", df.isnull().sum()[df.isnull().sum() > 0])
    print("Attrition rate: {:.1%}".format(df["AttritionFlag"].mean()))
    print("Avg estimated replacement cost: ${:,.0f}".format(df["EstReplacementCost"].mean()))
    df.to_csv(clean_path, index=False)
    print(f"\nCleaned data saved to {clean_path}")
    return df

if __name__ == "__main__":
    load_and_clean()