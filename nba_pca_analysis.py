"""
NBA Draft Combine — Factor Analysis / PCA
Luc-Alexandre Grenier & Patrick Reilly

Run from Anaconda Prompt:
    cd C:\Users\Luc\.cache\kagglehub\datasets\wyattowalsh\basketball\versions\231
    python nba_pca_analysis.py

Outputs saved to outputs/ and data/:
    outputs/regression_coef_plot.png
    outputs/cluster_selection.png
    outputs/archetype_clusters.png
    outputs/dark_horse_scatter.png
    outputs/bust_vs_darkhorse.png
    outputs/prospects_2026.png
    data/pca_factor_scores.csv
    data/prospects_2026_scores.csv
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 110

HERE     = Path(__file__).parent
CSV_DIR  = HERE / "csv"

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────
print("Loading CSVs...")
combine     = pd.read_csv(CSV_DIR / "draft_combine_stats.csv")
history     = pd.read_csv(CSV_DIR / "draft_history.csv")
player_info = pd.read_csv(CSV_DIR / "common_player_info.csv")
print(f"  combine: {combine.shape}  history: {history.shape}  player_info: {player_info.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. JOIN — only pull columns from history/player_info that combine does NOT
#    already have, to prevent pandas _x/_y suffix collisions.
# ─────────────────────────────────────────────────────────────────────────────
print("\nJoining tables...")
combine["player_id"]     = combine["player_id"].astype(str).str.strip()
history["person_id"]     = history["person_id"].astype(str).str.strip()
player_info["person_id"] = player_info["person_id"].astype(str).str.strip()

history_clean = (
    history[history["draft_type"] == "Draft"]
    .sort_values("overall_pick")
    .drop_duplicates(subset="person_id", keep="first")
)

# Columns already in combine that we must NOT bring from the right-hand tables:
#   season, position, first_name, last_name
df = combine.merge(
    history_clean[["person_id", "overall_pick", "round_number",
                   "round_pick", "organization_type"]],
    left_on="player_id", right_on="person_id",
    how="inner"
)

df = df.merge(
    player_info[["person_id", "from_year", "to_year",
                 "greatest_75_flag", "dleague_flag"]],
    on="person_id",
    how="inner"
)

# Guard against any accidental suffix columns
suffix_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
if suffix_cols:
    print(f"  WARNING — suffix columns found (duplicate clash): {suffix_cols}")
else:
    print(f"  OK — no _x/_y suffix columns. Shape: {df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. FILTER + FORCE NUMERIC TYPES
# ─────────────────────────────────────────────────────────────────────────────
print("\nFiltering to 2001–2019 draft classes...")
df["season"] = pd.to_numeric(df["season"], errors="coerce")
df = df[(df["season"] >= 2001) & (df["season"] <= 2019)].copy()
print(f"  Rows after filter: {len(df)}")

NUMERIC_COLS = [
    "height_wo_shoes", "weight", "wingspan", "standing_reach", "body_fat_pct",
    "standing_vertical_leap", "max_vertical_leap",
    "lane_agility_time", "three_quarter_sprint", "bench_press",
    "spot_fifteen_corner_left", "spot_fifteen_break_left", "spot_fifteen_top_key",
    "spot_fifteen_break_right", "spot_fifteen_corner_right",
    "spot_college_corner_left", "spot_college_break_left", "spot_college_top_key",
    "spot_college_break_right", "spot_college_corner_right",
    "overall_pick", "from_year", "to_year",
]

print("\nForcing numeric types (coercing non-numeric values to NaN)...")
for col in NUMERIC_COLS:
    if col not in df.columns:
        print(f"  MISSING: {col}")
        continue
    before = df[col].isna().sum()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    new_nans = df[col].isna().sum() - before
    if new_nans > 0:
        print(f"  {col:<40} {new_nans} non-numeric strings → NaN")

df["career_length"] = (
    pd.to_numeric(df["to_year"],   errors="coerce") -
    pd.to_numeric(df["from_year"], errors="coerce")
).clip(lower=0)

print(f"\n  career_length — mean: {df['career_length'].mean():.1f}, "
      f"nulls: {df['career_length'].isna().sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. SELECT COMBINE VARIABLES
# ─────────────────────────────────────────────────────────────────────────────
SIZE_VARS    = ["height_wo_shoes", "weight", "wingspan", "standing_reach", "body_fat_pct"]
ATH_VARS     = ["standing_vertical_leap", "max_vertical_leap",
                "lane_agility_time", "three_quarter_sprint", "bench_press"]
SHOOT15_VARS = ["spot_fifteen_corner_left", "spot_fifteen_break_left",
                "spot_fifteen_top_key", "spot_fifteen_break_right",
                "spot_fifteen_corner_right"]
SHOOTCOL_VARS= ["spot_college_corner_left", "spot_college_break_left",
                "spot_college_top_key", "spot_college_break_right",
                "spot_college_corner_right"]
ALL_20       = SIZE_VARS + ATH_VARS + SHOOT15_VARS + SHOOTCOL_VARS

EXTRA_COLS   = ["player_name", "career_length", "overall_pick",
                "organization_type", "position", "greatest_75_flag"]

missing_vars = [c for c in ALL_20 + EXTRA_COLS if c not in df.columns]
if missing_vars:
    print(f"\nWARNING — columns missing from df: {missing_vars}")
    raise SystemExit("Fix missing columns before continuing.")

# ─────────────────────────────────────────────────────────────────────────────
# 5. BUILD WORKING DATASET
# ─────────────────────────────────────────────────────────────────────────────
pca_df = df[ALL_20 + EXTRA_COLS].copy()

n_before = len(pca_df)
pca_df   = pca_df.dropna(subset=ALL_20, thresh=10)
print(f"\nDropped {n_before - len(pca_df)} players (< 10 drills recorded). "
      f"Remaining: {len(pca_df)}")

# ── Drop columns that are almost entirely NaN ─────────────────────────────
# SimpleImputer silently drops all-NaN columns, causing shape mismatches.
# We detect and remove sparse columns explicitly before imputing.
total      = len(pca_df)
non_null   = pca_df[ALL_20].notna().sum()
coverage   = non_null / total

print("\nCombine variable coverage (% of players with real data):")
USED_VARS, DROPPED_VARS = [], []
for col in ALL_20:
    pct  = coverage[col]
    bar  = "█" * int(pct * 20)
    keep = non_null[col] >= 30
    flag = "" if keep else "  ← DROPPED (too sparse)"
    print(f"  {pct:5.1%}  {bar:<20}  {col}{flag}")
    (USED_VARS if keep else DROPPED_VARS).append(col)

print(f"\nUsing {len(USED_VARS)} variables for PCA.")
if DROPPED_VARS:
    print(f"Dropped: {DROPPED_VARS}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. IMPUTE + SCALE
# ─────────────────────────────────────────────────────────────────────────────
imputer      = SimpleImputer(strategy="median")
X_imputed    = imputer.fit_transform(pca_df[USED_VARS])
X_imputed_df = pd.DataFrame(X_imputed, columns=USED_VARS, index=pca_df.index)

assert X_imputed_df.shape[1] == len(USED_VARS), \
    f"Shape mismatch after imputation: {X_imputed_df.shape[1]} cols vs {len(USED_VARS)} expected"
print(f"\nImputed shape: {X_imputed_df.shape}  — NaNs remaining: {X_imputed_df.isna().any().any()}")

scaler      = StandardScaler()
X_scaled    = scaler.fit_transform(X_imputed_df)
X_scaled_df = pd.DataFrame(X_scaled, columns=USED_VARS, index=pca_df.index)
print("Scaled — means:", X_scaled_df.mean().round(3).tolist())

# ─────────────────────────────────────────────────────────────────────────────
# 7. PCA — ALL COMPONENTS (for scree plot)
# ─────────────────────────────────────────────────────────────────────────────
n_vars      = len(USED_VARS)
pca_full    = PCA(n_components=n_vars, random_state=42)
pca_full.fit(X_scaled)

eigenvalues = pca_full.explained_variance_
var_ratio   = pca_full.explained_variance_ratio_
cum_var     = np.cumsum(var_ratio)

print("\nVariance explained per component:")
print(f"  {'PC':<6} {'Eigenvalue':>10} {'Var%':>8} {'Cum%':>8}")
for i in range(n_vars):
    print(f"  PC{i+1:<4} {eigenvalues[i]:>10.3f} {var_ratio[i]*100:>7.2f}% {cum_var[i]*100:>7.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 8. SCREE PLOT
# ─────────────────────────────────────────────────────────────────────────────
pc_labels = [f"PC{i+1}" for i in range(n_vars)]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(pc_labels, eigenvalues, "o-", color="steelblue", lw=2, markersize=7)
axes[0].axhline(1, color="red", linestyle="--", lw=1.5, label="Kaiser = 1")
axes[0].set_title("Scree Plot — Look for the elbow")
axes[0].set_xlabel("Component"); axes[0].set_ylabel("Eigenvalue")
axes[0].tick_params(axis="x", rotation=45); axes[0].legend()
for i, ev in enumerate(eigenvalues[:8]):
    axes[0].annotate(f"{ev:.2f}", (pc_labels[i], ev),
                     textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8)

axes[1].bar(pc_labels, var_ratio * 100, color="steelblue", alpha=0.6, label="Individual %")
axes[1].plot(pc_labels, cum_var * 100, "o-", color="darkred", lw=2, markersize=6, label="Cumulative %")
axes[1].axhline(70, color="green",  linestyle="--", lw=1.5, label="70%")
axes[1].axhline(80, color="orange", linestyle="--", lw=1.5, label="80%")
axes[1].set_title("Cumulative Variance Explained")
axes[1].set_xlabel("Component"); axes[1].set_ylabel("Variance (%)")
axes[1].tick_params(axis="x", rotation=45); axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(HERE / "scree_plot.png", bbox_inches="tight")
plt.show()
print("Saved: scree_plot.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9. CHOOSE N COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
n_kaiser     = int((eigenvalues > 1).sum())
n_70         = int(np.argmax(cum_var >= 0.70)) + 1
n_80         = int(np.argmax(cum_var >= 0.80)) + 1
N_COMPONENTS = n_kaiser   # change this number if the scree elbow suggests differently

print(f"\nKaiser criterion: {n_kaiser} components")
print(f"To reach 70%:     {n_70} components")
print(f"To reach 80%:     {n_80} components")
print(f"Using:            {N_COMPONENTS} components")

# ─────────────────────────────────────────────────────────────────────────────
# 10. FINAL PCA + FACTOR SCORES
# ─────────────────────────────────────────────────────────────────────────────
pca    = PCA(n_components=N_COMPONENTS, random_state=42)
scores = pca.fit_transform(X_scaled)

pc_cols   = [f"PC{i+1}" for i in range(N_COMPONENTS)]
scores_df = pd.DataFrame(scores, columns=pc_cols, index=pca_df.index)
scores_df["player_name"]       = pca_df["player_name"].values
scores_df["career_length"]     = pca_df["career_length"].values
scores_df["overall_pick"]      = pca_df["overall_pick"].values
scores_df["greatest_75"]       = pca_df["greatest_75_flag"].values
scores_df["organization_type"] = pca_df["organization_type"].values
scores_df["position"]          = pca_df["position"].values

print(f"\nFactor scores shape: {scores_df[pc_cols].shape}")
print(f"Total variance retained: {pca.explained_variance_ratio_.sum():.1%}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. LOADINGS MATRIX
# ─────────────────────────────────────────────────────────────────────────────
loadings = pd.DataFrame(
    pca.components_.T,
    index=USED_VARS,
    columns=pc_cols
)

print("\nLoadings matrix:")
print(loadings.round(3).to_string())

print("\nTop loadings per component:")
for pc in pc_cols:
    ranked = loadings[pc].abs().sort_values(ascending=False)
    pct    = pca.explained_variance_ratio_[pc_cols.index(pc)] * 100
    print(f"\n  {pc} ({pct:.1f}% variance):")
    for var in ranked.index[:6]:
        val  = loadings.loc[var, pc]
        bar  = "█" * int(abs(val) * 15)
        sign = "+" if val > 0 else "-"
        flag = "  ← STRONG" if abs(val) > 0.45 else ""
        print(f"    {sign}{abs(val):.3f}  {bar:<12}  {var}{flag}")

# ── Loadings heatmap ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(max(8, N_COMPONENTS * 1.5), max(5, len(USED_VARS) * 0.45)))
sns.heatmap(loadings, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax)

# Group dividers (draw between size/ath/shooting groups)
used_set = set(USED_VARS)
running  = 0
for grp in [SIZE_VARS, ATH_VARS, SHOOT15_VARS]:
    running += len([v for v in grp if v in used_set])
    if 0 < running < len(USED_VARS):
        ax.axhline(running, color="black", lw=2)

ax.set_title("PCA Loadings Heatmap  |  Red = positive, Blue = negative", fontsize=11)
ax.set_xlabel("Principal Component"); ax.set_ylabel("Variable")
plt.tight_layout()
plt.savefig(HERE / "loadings_heatmap.png", bbox_inches="tight")
plt.show()
print("Saved: loadings_heatmap.png")

# ── Name components (edit these after reading the loadings above) ─────────
COMPONENT_NAMES = {
    "PC1": "Size",
    "PC2": "Athleticism",
    "PC3": "Shooting Readiness",
    "PC4": "Strength",
    "PC5": "Shooting Range",
}
COMPONENT_NAMES = {k: v for k, v in COMPONENT_NAMES.items() if k in pc_cols}
print("\nComponent names:", COMPONENT_NAMES)

# ─────────────────────────────────────────────────────────────────────────────
# 12. BIPLOT (PC1 vs PC2)
# ─────────────────────────────────────────────────────────────────────────────
def career_colour(yrs):
    if pd.isna(yrs): return "lightgrey"
    elif yrs <  3:   return "salmon"
    elif yrs <  8:   return "gold"
    elif yrs < 13:   return "steelblue"
    else:            return "darkgreen"

colours = [career_colour(y) for y in scores_df["career_length"]]
fig, ax = plt.subplots(figsize=(11, 8))
ax.scatter(scores_df["PC1"], scores_df["PC2"], c=colours, alpha=0.5, s=25, zorder=2)

for _, row in scores_df[scores_df["greatest_75"] == "Y"].iterrows():
    ax.annotate(row["player_name"].split()[-1], (row["PC1"], row["PC2"]),
                fontsize=7, color="darkgreen", fontweight="bold",
                xytext=(4, 4), textcoords="offset points")

grp_colours = {"size":"darkred","ath":"navy","shoot15":"darkorange","shootcol":"purple"}
var_groups  = ([(v,"size") for v in SIZE_VARS if v in used_set] +
               [(v,"ath")  for v in ATH_VARS  if v in used_set] +
               [(v,"shoot15")  for v in SHOOT15_VARS  if v in used_set] +
               [(v,"shootcol") for v in SHOOTCOL_VARS if v in used_set])
scale = 3.5
for var, grp in var_groups:
    xl = loadings.loc[var, "PC1"] * scale
    yl = loadings.loc[var, "PC2"] * scale
    ax.annotate("", xy=(xl, yl), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=grp_colours[grp], lw=1.4))
    if np.sqrt(xl**2 + yl**2) > 1.2:
        lbl = var.replace("spot_fifteen_","15_").replace("spot_college_","col_")
        ax.text(xl*1.07, yl*1.07, lbl, fontsize=7, color=grp_colours[grp], ha="center")

pc1_name = COMPONENT_NAMES.get("PC1", "PC1")
pc2_name = COMPONENT_NAMES.get("PC2", "PC2")
ax.set_xlabel(f"PC1 — {pc1_name} ({pca.explained_variance_ratio_[0]:.1%})", fontsize=11)
ax.set_ylabel(f"PC2 — {pc2_name} ({pca.explained_variance_ratio_[1]:.1%})", fontsize=11)
ax.set_title("Biplot — Player scores (dots) + Variable loadings (arrows)", fontsize=12)
ax.axhline(0, color="grey", lw=0.7, ls="--")
ax.axvline(0, color="grey", lw=0.7, ls="--")
ax.legend(handles=[
    mpatches.Patch(color="salmon",    label="Bust (< 3 yrs)"),
    mpatches.Patch(color="gold",      label="Role Player (3–7 yrs)"),
    mpatches.Patch(color="steelblue", label="Starter (8–12 yrs)"),
    mpatches.Patch(color="darkgreen", label="Star (13+ yrs)"),
], fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig(HERE / "biplot.png", bbox_inches="tight")
plt.show()
print("Saved: biplot.png")

# ─────────────────────────────────────────────────────────────────────────────
# 13. SANITY CHECK — SCORES BY POSITION
# ─────────────────────────────────────────────────────────────────────────────
def simplify_pos(pos):
    if pd.isna(pos): return "Unknown"
    pos = str(pos)
    if "Center" in pos:                            return "Center"
    elif "Forward" in pos and "Guard" not in pos:  return "Forward"
    elif "Guard"   in pos and "Forward" not in pos:return "Guard"
    else:                                          return "Fwd-Guard"

scores_df["pos_simple"] = scores_df["position"].apply(simplify_pos)
pos_order   = ["Guard","Fwd-Guard","Forward","Center"]
pos_palette = {"Guard":"steelblue","Fwd-Guard":"mediumseagreen",
               "Forward":"gold","Center":"salmon"}

n_plots = min(N_COMPONENTS, 3)
fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
if n_plots == 1: axes = [axes]
for i, pc in enumerate(pc_cols[:3]):
    plot_data = scores_df[scores_df["pos_simple"].isin(pos_order)]
    sns.boxplot(data=plot_data, x="pos_simple", y=pc, order=pos_order,
                palette=pos_palette, ax=axes[i], width=0.5)
    axes[i].set_title(f"{pc}: \"{COMPONENT_NAMES.get(pc, pc)}\"")
    axes[i].set_xlabel("")
    axes[i].axhline(0, color="black", lw=0.8, ls="--")
    axes[i].tick_params(axis="x", rotation=20)
plt.suptitle("Factor Scores by Position (sanity check)", fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(HERE / "scores_by_position.png", bbox_inches="tight")
plt.show()
print("Saved: scores_by_position.png")

# ─────────────────────────────────────────────────────────────────────────────
# 14. EXPORT FACTOR SCORES FOR REGRESSION
# ─────────────────────────────────────────────────────────────────────────────
export = (scores_df[pc_cols + ["player_name","career_length","overall_pick",
                                "organization_type","position","greatest_75"]]
          .rename(columns=COMPONENT_NAMES))

out_path = HERE / "pca_factor_scores.csv"
export.to_csv(out_path, index=False)
print(f"\nSaved: pca_factor_scores.csv  ({export.shape[0]} rows, {export.shape[1]} cols)")
print(export.head())

print("\n✓ PCA analysis complete.")
