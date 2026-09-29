# Week 3 - Statistical Analysis and Hypothesis Testing

## Netflix Movies and TV Shows Dataset

---

## Project Overview

This project focuses on statistical analysis and hypothesis testing using the Netflix Movies and TV Shows dataset.

The main objective is to formulate testable hypotheses, apply appropriate statistical tests using Python, interpret p-values and confidence intervals, and communicate the results through statistical visualizations.

This project is a continuation of the previous Netflix data analysis projects:

- Week 1: Data Cleaning, Preprocessing and EDA
- Week 2: Advanced Data Visualization and Storytelling
- Week 3: Statistical Analysis and Hypothesis Testing

---

## Objectives

The main objectives of this project are:

- Analyze the Netflix Movies and TV Shows dataset.
- Formulate statistically testable hypotheses.
- Compare Movies and TV Shows using statistical methods.
- Perform an independent two-sample t-test.
- Perform a Chi-square test of independence.
- Calculate a 95% confidence interval.
- Calculate an effect size using Cohen's d.
- Interpret statistical significance using p-values.
- Create visualizations supporting the statistical analysis.
- Document the findings and conclusions.

---

## Dataset

### Dataset Name

Netflix Movies and TV Shows

### Dataset Format

CSV

### Important Columns

| Column | Description |
|---|---|
| show_id | Unique identifier for each title |
| type | Movie or TV Show |
| title | Name of the title |
| director | Director of the title |
| cast | Cast members |
| country | Country or countries associated with the title |
| date_added | Date added to Netflix |
| release_year | Original release year |
| rating | Content rating |
| duration | Movie runtime or number of TV seasons |
| listed_in | Genres or categories |
| description | Description of the title |

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy

---

## Statistical Methods

Two major statistical tests are performed.

### 1. Independent Two-Sample t-Test

The first hypothesis investigates whether the average release year of Movies is different from the average release year of TV Shows.

#### Null Hypothesis (H0)

There is no difference between the mean release years of Movies and TV Shows.

```text
H0: μMovies = μTVShows