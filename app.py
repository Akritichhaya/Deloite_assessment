import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import io

st.set_page_config(
    page_title="ML Data Analysis & Visualization",
    page_icon="📊",
    layout="wide"
)

st.title("📊 ML Data Analysis & Visualization")
st.markdown("Load a dataset, preprocess it, apply **Linear Regression**, and explore results through interactive charts.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    dataset_option = st.selectbox(
        "Dataset",
        ["California Housing", "Diabetes", "Upload CSV"],
        help="Choose a built-in dataset or upload your own CSV file."
    )

    uploaded_file = None
    if dataset_option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    st.markdown("---")
    test_size = st.slider("Test set size (%)", 10, 40, 20, step=5) / 100
    random_seed = st.number_input("Random seed", value=42, min_value=0, max_value=9999)
    scale_features = st.checkbox("Scale features (StandardScaler)", value=True)

    st.markdown("---")
    st.markdown("**About**")
    st.caption("This app demonstrates a full ML workflow: data loading → preprocessing → linear regression → evaluation & visualization.")


# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_builtin(name):
    if name == "California Housing":
        data = fetch_california_housing(as_frame=True)
        df = data.frame.copy()
        target_col = "MedHouseVal"
        description = (
            "**California Housing** — Median house prices across California districts.\n\n"
            "Target: `MedHouseVal` (median house value in $100,000s)."
        )
    else:
        data = load_diabetes(as_frame=True)
        df = data.frame.copy()
        target_col = "target"
        description = (
            "**Diabetes** — Disease progression dataset with 10 baseline features.\n\n"
            "Target: `target` (quantitative measure of disease progression one year after baseline)."
        )
    return df, target_col, description


def load_csv(file):
    df = pd.read_csv(file)
    return df


# ── Load dataset ──────────────────────────────────────────────────────────────
if dataset_option == "Upload CSV":
    if uploaded_file is None:
        st.info("⬆️ Please upload a CSV file using the sidebar to get started.")
        st.stop()
    df = load_csv(uploaded_file)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("The uploaded CSV must have at least 2 numeric columns.")
        st.stop()
    target_col = st.sidebar.selectbox("Target column", numeric_cols, index=len(numeric_cols) - 1)
    description = f"Custom CSV with {df.shape[0]} rows and {df.shape[1]} columns. Target: `{target_col}`."
else:
    df, target_col, description = load_builtin(dataset_option)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🗂️ Dataset Overview",
    "🔬 Preprocessing",
    "🤖 Model & Evaluation",
    "📈 Visualizations"
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dataset Overview
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Dataset Overview")
    st.markdown(description)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing values", int(df.isnull().sum().sum()))
    col4.metric("Target", target_col)

    st.markdown("#### Sample rows")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("#### Descriptive statistics")
    st.dataframe(df.describe().round(3), use_container_width=True)

    st.markdown("#### Missing values per column")
    missing = df.isnull().sum().rename("Missing").to_frame()
    missing["% Missing"] = (missing["Missing"] / len(df) * 100).round(2)
    missing = missing[missing["Missing"] > 0]
    if missing.empty:
        st.success("No missing values found.")
    else:
        st.dataframe(missing, use_container_width=True)

    st.markdown("#### Target distribution")
    fig = px.histogram(
        df, x=target_col, nbins=50, marginal="box",
        title=f"Distribution of {target_col}",
        color_discrete_sequence=["#636EFA"]
    )
    fig.update_layout(bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Preprocessing
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Preprocessing")

    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    selected_features = st.multiselect(
        "Select feature columns to include",
        options=feature_cols,
        default=feature_cols,
        help="Choose which numeric features to use for training the model."
    )

    if not selected_features:
        st.warning("Please select at least one feature column.")
        st.stop()

    X_raw = df[selected_features].copy()
    y_raw = df[target_col].copy()

    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(imputer.fit_transform(X_raw), columns=selected_features)
    y_clean = y_raw.fillna(y_raw.median())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y_clean, test_size=test_size, random_state=int(random_seed)
    )

    # Scaling
    if scale_features:
        scaler = StandardScaler()
        X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=selected_features)
        X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=selected_features)
    else:
        X_train_sc, X_test_sc = X_train.copy(), X_test.copy()

    col1, col2, col3 = st.columns(3)
    col1.metric("Training samples", len(X_train))
    col2.metric("Test samples", len(X_test))
    col3.metric("Features", len(selected_features))

    st.markdown("#### Feature correlation with target")
    corr_vals = X_imputed.assign(**{target_col: y_clean}).corr()[target_col].drop(target_col).sort_values()
    fig = px.bar(
        x=corr_vals.values, y=corr_vals.index, orientation="h",
        color=corr_vals.values, color_continuous_scale="RdBu",
        title=f"Pearson correlation with {target_col}",
        labels={"x": "Correlation", "y": "Feature"}
    )
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Full correlation heatmap")
    corr_matrix = X_imputed.assign(**{target_col: y_clean}).corr().round(2)
    fig_heat = px.imshow(
        corr_matrix, text_auto=True, color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto",
        title="Correlation Matrix"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("#### Feature distributions")
    n_feat = len(selected_features)
    ncols = 3
    nrows = (n_feat + ncols - 1) // ncols
    fig_dist, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.array(axes).flatten()
    for i, feat in enumerate(selected_features):
        axes[i].hist(X_imputed[feat].dropna(), bins=40, color="#636EFA", alpha=0.8, edgecolor="none")
        axes[i].set_title(feat, fontsize=10)
        axes[i].tick_params(labelsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig_dist.tight_layout()
    st.pyplot(fig_dist)
    plt.close(fig_dist)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model & Evaluation
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Linear Regression — Model & Evaluation")

    model = LinearRegression()
    model.fit(X_train_sc, y_train)

    y_pred_train = model.predict(X_train_sc)
    y_pred_test = model.predict(X_test_sc)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    st.markdown("#### Performance metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("R² (train)", f"{r2_train:.4f}")
    m2.metric("R² (test)", f"{r2_test:.4f}")
    m3.metric("RMSE (test)", f"{rmse_test:.4f}")
    m4.metric("MSE (test)", f"{mse_test:.4f}")
    m5.metric("MAE (test)", f"{mae_test:.4f}")

    st.markdown("""
    | Metric | Meaning |
    |--------|---------|
    | **R²** | Proportion of variance explained (1.0 = perfect) |
    | **RMSE** | Root mean squared error — penalises large errors |
    | **MSE** | Mean squared error |
    | **MAE** | Mean absolute error — average prediction error |
    """)

    st.markdown("#### Model coefficients")
    coef_df = pd.DataFrame({
        "Feature": selected_features,
        "Coefficient": model.coef_
    }).sort_values("Coefficient", key=abs, ascending=False).reset_index(drop=True)
    coef_df["Intercept"] = ""
    coef_df.loc[len(coef_df)] = {"Feature": "(Intercept)", "Coefficient": model.intercept_, "Intercept": "✓"}
    st.dataframe(coef_df[["Feature", "Coefficient"]], use_container_width=True)

    st.markdown("#### Predictions vs Actual (test set sample)")
    sample_size = min(200, len(y_test))
    sample_idx = np.random.RandomState(int(random_seed)).choice(len(y_test), sample_size, replace=False)
    preview_df = pd.DataFrame({
        "Actual": np.array(y_test)[sample_idx],
        "Predicted": y_pred_test[sample_idx],
    })
    st.dataframe(preview_df.round(4).reset_index(drop=True), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Visualizations
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Visualizations")

    # Actual vs Predicted
    st.markdown("#### Actual vs Predicted")
    fig_avp = px.scatter(
        x=y_test, y=y_pred_test,
        labels={"x": f"Actual {target_col}", "y": f"Predicted {target_col}"},
        title="Actual vs Predicted (test set)",
        color_discrete_sequence=["#636EFA"],
        opacity=0.6,
        trendline="ols"
    )
    perfect = [float(min(y_test.min(), y_pred_test.min())), float(max(y_test.max(), y_pred_test.max()))]
    fig_avp.add_trace(go.Scatter(x=perfect, y=perfect, mode="lines", name="Perfect fit",
                                  line=dict(color="red", dash="dash")))
    fig_avp.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_avp, use_container_width=True)

    # Residuals
    st.markdown("#### Residuals plot")
    residuals = np.array(y_test) - y_pred_test
    fig_res = make_subplots(rows=1, cols=2,
                            subplot_titles=["Residuals vs Predicted", "Residual distribution"])
    fig_res.add_trace(go.Scatter(x=y_pred_test, y=residuals, mode="markers",
                                  marker=dict(color="#EF553B", opacity=0.5, size=4),
                                  name="Residuals"), row=1, col=1)
    fig_res.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)
    fig_res.add_trace(go.Histogram(x=residuals, nbinsx=50,
                                    marker_color="#636EFA", opacity=0.8,
                                    name="Distribution"), row=1, col=2)
    fig_res.update_layout(showlegend=False, title="Residual Analysis")
    st.plotly_chart(fig_res, use_container_width=True)

    # Feature importance (by absolute coefficient value)
    st.markdown("#### Feature importance (|coefficient|)")
    imp_df = pd.DataFrame({
        "Feature": selected_features,
        "Importance": np.abs(model.coef_)
    }).sort_values("Importance", ascending=True)
    fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                      color="Importance", color_continuous_scale="Blues",
                      title="Feature Importance by |Coefficient|")
    fig_imp.update_coloraxes(showscale=False)
    st.plotly_chart(fig_imp, use_container_width=True)

    # Learning curve
    st.markdown("#### Learning curve")
    with st.spinner("Computing learning curve…"):
        train_sizes, train_scores, val_scores = learning_curve(
            LinearRegression(), X_train_sc, y_train,
            cv=5, scoring="r2",
            train_sizes=np.linspace(0.1, 1.0, 8),
            n_jobs=-1
        )
    fig_lc = go.Figure()
    fig_lc.add_trace(go.Scatter(
        x=train_sizes, y=train_scores.mean(axis=1),
        error_y=dict(array=train_scores.std(axis=1)),
        mode="lines+markers", name="Train R²", line=dict(color="#636EFA")
    ))
    fig_lc.add_trace(go.Scatter(
        x=train_sizes, y=val_scores.mean(axis=1),
        error_y=dict(array=val_scores.std(axis=1)),
        mode="lines+markers", name="CV R²", line=dict(color="#EF553B")
    ))
    fig_lc.update_layout(
        title="Learning Curve (R² vs training size)",
        xaxis_title="Training samples", yaxis_title="R²"
    )
    st.plotly_chart(fig_lc, use_container_width=True)

    # Pairplot (top correlated features)
    st.markdown("#### Scatter matrix (top features vs target)")
    top_feats = coef_df[coef_df["Feature"] != "(Intercept)"]["Feature"].head(4).tolist()
    plot_cols = top_feats + [target_col]
    pair_df = X_imputed[top_feats].assign(**{target_col: y_clean.values}).sample(
        min(500, len(X_imputed)), random_state=int(random_seed)
    )
    fig_pair = px.scatter_matrix(
        pair_df, dimensions=plot_cols,
        color=target_col, color_continuous_scale="Viridis",
        title="Scatter Matrix (top 4 features + target)"
    )
    fig_pair.update_traces(marker=dict(size=3, opacity=0.5), diagonal_visible=False)
    st.plotly_chart(fig_pair, use_container_width=True)

st.markdown("---")
st.caption("Built with Streamlit · scikit-learn · Plotly · Pandas · NumPy")
