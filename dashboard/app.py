import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

st.set_page_config(page_title="HR Attrition Analytics", layout="wide")
sns.set_style("whitegrid")

# --- Load data ---
@st.cache_data
def load_data():
    return pd.read_csv("data/hr_clean.csv")

df = load_data()

st.title("Employee Attrition Analytics")
st.caption("Who's leaving, why, and what it costs — built on the IBM HR Analytics dataset")

# --- Sidebar filter ---
departments = st.sidebar.multiselect(
    "Filter by department",
    options=df["Department"].unique(),
    default=df["Department"].unique(),
)

# --- Recreate personas (same logic as the notebook) ---
@st.cache_data
def add_personas(data):
    data = data.copy()
    data["OverTimeFlag"] = (data["OverTime"] == "Yes").astype(int)

    cluster_cols = ["MonthlyIncome", "YearsAtCompany", "JobSatisfaction", "WorkLifeBalance", "OverTimeFlag"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[cluster_cols])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    data["Cluster"] = kmeans.fit_predict(X_scaled)

    # Map clusters to names based on the profile we found in the notebook
    profile = data.groupby("Cluster")["AttritionFlag"].mean().sort_values()
    ordered_clusters = profile.index.tolist()
    persona_names = {
        ordered_clusters[0]: "Senior Veteran",
        ordered_clusters[1]: "Stable Mid-Career",
        ordered_clusters[2]: "Dissatisfied Steady-Stater",
        ordered_clusters[3]: "Overworked & At-Risk",
    }
    data["Persona"] = data["Cluster"].map(persona_names)
    return data

df = add_personas(df)


filtered = df[df["Department"].isin(departments)]

# --- KPI row ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Headcount", len(filtered))
col2.metric("Attrition rate", f"{filtered['AttritionFlag'].mean():.1%}")
col3.metric("Avg. monthly income", f"£{filtered['MonthlyIncome'].mean():,.0f}")
col4.metric("Avg. replacement cost", f"£{filtered['EstReplacementCost'].mean():,.0f}")

st.divider()

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Attrition rate by department")
    dept_rate = filtered.groupby("Department")["AttritionFlag"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x=dept_rate.values, y=dept_rate.index, ax=ax, color="#c0392b")
    ax.set_xlabel("Attrition rate")
    st.pyplot(fig)

with col_b:
    st.subheader("Attrition rate by overtime status")
    ot_rate = filtered.groupby("OverTime")["AttritionFlag"].mean()
    fig2, ax2 = plt.subplots()
    sns.barplot(x=ot_rate.index, y=ot_rate.values, ax=ax2, color="#2c3e50")
    ax2.set_ylabel("Attrition rate")
    st.pyplot(fig2)

st.divider()
st.subheader("Employee personas: risk and cost")

persona_summary = filtered_personas = df[df["Department"].isin(departments)].groupby("Persona").agg(
    Headcount=("Persona", "count"),
    AttritionRate=("AttritionFlag", "mean"),
    AvgReplacementCost=("EstReplacementCost", "mean"),
).round(2).sort_values("AttritionRate", ascending=False)

col_c, col_d = st.columns(2)

with col_c:
    display_df = persona_summary.copy()
    display_df["AttritionRate"] = display_df["AttritionRate"].apply(lambda x: f"{x:.1%}")
    display_df["AvgReplacementCost"] = display_df["AvgReplacementCost"].apply(lambda x: f"£{x:,.0f}")
    st.markdown(display_df.to_html(), unsafe_allow_html=True)

with col_d:
    total_cost_by_persona = df[(df["Department"].isin(departments)) & (df["AttritionFlag"] == 1)] \
        .groupby("Persona")["EstReplacementCost"].sum().sort_values(ascending=False)
    fig3, ax3 = plt.subplots()
    sns.barplot(x=total_cost_by_persona.values, y=total_cost_by_persona.index, ax=ax3, color="#16a085")
    ax3.set_xlabel("Total replacement cost (£)")
    st.pyplot(fig3)

st.divider()
st.subheader("Predicted flight risk — current employees")

@st.cache_data
def get_risk_scores(data):
    data = data.copy()
    drop_cols = ["Attrition", "AttritionFlag", "EmployeeNumber", "Cluster", "Persona",
                 "OverTimeFlag", "EstAnnualSalary", "ReplacementMultiplier", "EstReplacementCost"]
    X = data.drop(columns=[c for c in drop_cols if c in data.columns])
    y = data["AttritionFlag"]

    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipeline.fit(X, y)
    data["RiskScore"] = pipeline.predict_proba(X)[:, 1]
    return data

df_with_risk = get_risk_scores(df)

# Only show people still at the company - no point flagging people who already left
current_employees = df_with_risk[df_with_risk["AttritionFlag"] == 0]
current_filtered = current_employees[current_employees["Department"].isin(departments)]

top_risk = current_filtered.sort_values("RiskScore", ascending=False).head(15)

display_cols = ["Department", "JobRole", "Persona", "MonthlyIncome", "OverTime", "JobSatisfaction", "RiskScore"]
display_risk = top_risk[display_cols].copy()
display_risk["RiskScore"] = display_risk["RiskScore"].apply(lambda x: f"{x:.0%}")
display_risk["MonthlyIncome"] = display_risk["MonthlyIncome"].apply(lambda x: f"£{x:,.0f}")

st.caption("Top 15 current employees flagged as highest attrition risk by the model")
st.markdown(display_risk.to_html(index=False), unsafe_allow_html=True)