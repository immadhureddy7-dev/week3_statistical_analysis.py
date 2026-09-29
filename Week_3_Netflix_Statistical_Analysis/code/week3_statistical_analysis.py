import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from scipy import stats


# ============================================================
# 1. CREATE VISUALIZATION FOLDER
# ============================================================

os.makedirs("../visualizations", exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(
    "../data/netflix_titles.csv",
    encoding="latin1"
)

print("\n========================================")
print("DATASET LOADED")
print("========================================")


# ============================================================
# 3. INITIAL DATA INSPECTION
# ============================================================

print("\nFIRST 5 ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nDATA TYPES")
print(df.dtypes)

print("\nDATASET INFORMATION")
df.info()


# ============================================================
# 4. DATA CLEANING
# ============================================================

print("\n========================================")
print("DATA CLEANING")
print("========================================")


# Remove duplicate rows

duplicate_count = df.duplicated().sum()

print("\nDuplicate rows:", duplicate_count)

df = df.drop_duplicates()


# Handle categorical missing values

if "director" in df.columns:
    df["director"] = df["director"].fillna("Unknown")

if "country" in df.columns:
    df["country"] = df["country"].fillna("Unknown")

if "cast" in df.columns:
    df["cast"] = df["cast"].fillna("Not Available")


# Handle rating missing values

if df["rating"].isnull().sum() > 0:

    rating_mode = df["rating"].mode()[0]

    df["rating"] = df["rating"].fillna(
        rating_mode
    )


# Convert release year

df["release_year"] = pd.to_numeric(
    df["release_year"],
    errors="coerce"
)


# Remove rows where essential variables are missing

df = df.dropna(
    subset=[
        "type",
        "release_year"
    ]
)


print("\nMissing values after cleaning:")

print(
    df[
        [
            "type",
            "release_year",
            "rating"
        ]
    ].isnull().sum()
)


# ============================================================
# 5. DESCRIPTIVE STATISTICS
# ============================================================

print("\n========================================")
print("DESCRIPTIVE STATISTICS")
print("========================================")


print("\nOverall Release Year Statistics:")

print(
    df["release_year"].describe()
)


print("\nStatistics by Content Type:")

group_statistics = (
    df.groupby("type")["release_year"]
      .agg(
          [
              "count",
              "mean",
              "median",
              "std",
              "min",
              "max"
          ]
      )
)

print(group_statistics)


# ============================================================
# 6. HYPOTHESIS 1
# ============================================================

print("\n========================================")
print("HYPOTHESIS 1 - INDEPENDENT T-TEST")
print("========================================")


print(
    """
Research Question:

Is the mean release year of Movies different
from the mean release year of TV Shows?
"""
)


# Null hypothesis

print(
    "H0: The mean release year of Movies and TV Shows is equal."
)

# Alternative hypothesis

print(
    "H1: The mean release year of Movies and TV Shows is different."
)


# Separate groups

movies = df[
    df["type"] == "Movie"
]["release_year"]

tv_shows = df[
    df["type"] == "TV Show"
]["release_year"]


print("\nMovie count:", len(movies))

print("TV Show count:", len(tv_shows))


print("\nMovie mean release year:")

print(movies.mean())


print("\nTV Show mean release year:")

print(tv_shows.mean())


# ============================================================
# 7. CHECK NORMALITY
# ============================================================

print("\nNormality test:")

# Shapiro-Wilk is not appropriate for extremely large
# datasets because it becomes overly sensitive.
# We therefore use the large-sample t-test with Welch correction.


# ============================================================
# 8. WELCH'S INDEPENDENT T-TEST
# ============================================================

t_statistic, p_value = stats.ttest_ind(
    movies,
    tv_shows,
    equal_var=False
)


print("\nT-statistic:")
print(t_statistic)

print("\nP-value:")
print(p_value)


# ============================================================
# 9. CONFIDENCE INTERVAL FOR MEAN DIFFERENCE
# ============================================================

movie_mean = movies.mean()

tv_mean = tv_shows.mean()

mean_difference = movie_mean - tv_mean


movie_variance = movies.var(
    ddof=1
)

tv_variance = tv_shows.var(
    ddof=1
)

movie_n = len(movies)

tv_n = len(tv_shows)


standard_error = np.sqrt(
    (movie_variance / movie_n)
    +
    (tv_variance / tv_n)
)


# Welch-Satterthwaite degrees of freedom

numerator = (
    (movie_variance / movie_n)
    +
    (tv_variance / tv_n)
) ** 2


denominator = (
    ((movie_variance / movie_n) ** 2)
    / (movie_n - 1)
    +
    ((tv_variance / tv_n) ** 2)
    / (tv_n - 1)
)


degrees_of_freedom = (
    numerator / denominator
)


critical_value = stats.t.ppf(
    0.975,
    degrees_of_freedom
)


margin_of_error = (
    critical_value
    *
    standard_error
)


ci_lower = (
    mean_difference
    -
    margin_of_error
)


ci_upper = (
    mean_difference
    +
    margin_of_error
)


print("\nMean difference (Movie - TV Show):")

print(mean_difference)


print("\n95% Confidence Interval:")

print(
    "(",
    ci_lower,
    ",",
    ci_upper,
    ")"
)


# ============================================================
# 10. EFFECT SIZE - COHEN'S D
# ============================================================

pooled_standard_deviation = np.sqrt(
    (
        ((movie_n - 1) * movie_variance)
        +
        ((tv_n - 1) * tv_variance)
    )
    /
    (
        movie_n
        +
        tv_n
        -
        2
    )
)


cohens_d = (
    mean_difference
    /
    pooled_standard_deviation
)


print("\nCohen's d:")

print(cohens_d)


# ============================================================
# 11. T-TEST DECISION
# ============================================================

alpha = 0.05

print("\nSignificance level:", alpha)


if p_value < alpha:

    print(
        "\nDecision: Reject the null hypothesis."
    )

    print(
        "There is statistically significant evidence "
        "that the mean release years differ."
    )

else:

    print(
        "\nDecision: Fail to reject the null hypothesis."
    )

    print(
        "There is insufficient statistical evidence "
        "that the mean release years differ."
    )


# ============================================================
# 12. VISUALIZATION - RELEASE YEAR DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

sns.histplot(
    data=df,
    x="release_year",
    hue="type",
    bins=30,
    kde=True,
    element="step"
)

plt.title(
    "Release Year Distribution by Content Type"
)

plt.xlabel(
    "Release Year"
)

plt.ylabel(
    "Number of Titles"
)

plt.tight_layout()

plt.savefig(
    "../visualizations/01_release_year_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# 13. VISUALIZATION - BOX PLOT
# ============================================================

plt.figure(
    figsize=(8, 6)
)

sns.boxplot(
    data=df,
    x="type",
    y="release_year"
)

plt.title(
    "Release Year Comparison: Movies vs TV Shows"
)

plt.xlabel(
    "Content Type"
)

plt.ylabel(
    "Release Year"
)

plt.tight_layout()

plt.savefig(
    "../visualizations/02_release_year_boxplot.png",
    dpi=300
)

plt.show()


# ============================================================
# 14. HYPOTHESIS 2
# ============================================================

print("\n========================================")
print("HYPOTHESIS 2 - CHI-SQUARE TEST")
print("========================================")


print(
    """
Research Question:

Is there a statistically significant association
between Netflix content type and content rating?
"""
)


print(
    "H0: Content type and rating are independent."
)

print(
    "H1: Content type and rating are associated."
)


# ============================================================
# 15. CREATE CONTINGENCY TABLE
# ============================================================

rating_table = pd.crosstab(
    df["type"],
    df["rating"]
)


print("\nContingency Table:")

print(rating_table)


# ============================================================
# 16. CHI-SQUARE TEST
# ============================================================

chi2, chi_p_value, dof, expected = (
    stats.chi2_contingency(
        rating_table
    )
)


print("\nChi-square statistic:")

print(chi2)


print("\nP-value:")

print(chi_p_value)


print("\nDegrees of freedom:")

print(dof)


# ============================================================
# 17. CHI-SQUARE DECISION
# ============================================================

if chi_p_value < alpha:

    print(
        "\nDecision: Reject the null hypothesis."
    )

    print(
        "There is statistically significant evidence "
        "of an association between content type and rating."
    )

else:

    print(
        "\nDecision: Fail to reject the null hypothesis."
    )

    print(
        "There is insufficient evidence of an association "
        "between content type and rating."
    )


# ============================================================
# 18. CHI-SQUARE VISUALIZATION
# ============================================================

plt.figure(
    figsize=(12, 6)
)

sns.heatmap(
    rating_table,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "Content Type vs Rating"
)

plt.xlabel(
    "Rating"
)

plt.ylabel(
    "Content Type"
)

plt.tight_layout()

plt.savefig(
    "../visualizations/03_type_rating_heatmap.png",
    dpi=300
)

plt.show()


# ============================================================
# 19. RATING DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(12, 6)
)

sns.countplot(
    data=df,
    x="rating",
    hue="type",
    order=df["rating"].value_counts().index
)

plt.title(
    "Rating Distribution by Content Type"
)

plt.xlabel(
    "Rating"
)

plt.ylabel(
    "Number of Titles"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    "../visualizations/04_rating_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# 20. SAVE STATISTICAL RESULTS
# ============================================================

results = pd.DataFrame(
    {
        "Test": [
            "Welch Independent T-Test",
            "Chi-Square Test"
        ],

        "Statistic": [
            t_statistic,
            chi2
        ],

        "P_Value": [
            p_value,
            chi_p_value
        ],

        "Alpha": [
            alpha,
            alpha
        ]
    }
)


results.to_csv(
    "../data/statistical_results.csv",
    index=False
)


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n========================================")
print("FINAL STATISTICAL SUMMARY")
print("========================================")


print(
    "\nT-Test statistic:",
    t_statistic
)

print(
    "T-Test p-value:",
    p_value
)

print(
    "Mean difference:",
    mean_difference
)

print(
    "95% CI:",
    ci_lower,
    "to",
    ci_upper
)

print(
    "Cohen's d:",
    cohens_d
)


print(
    "\nChi-square statistic:",
    chi2
)

print(
    "Chi-square p-value:",
    chi_p_value
)


print("\n========================================")
print("WEEK 3 ANALYSIS COMPLETED")
print("========================================")