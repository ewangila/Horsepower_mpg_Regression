# mtcars: MPG vs Horsepower — OLS Inference & Diagnostics

An end-to-end Ordinary Least Squares (OLS) regression analysis of the relationship between **miles per gallon (mpg)** and **horsepower (hp)** using the classic `mtcars` dataset.

The project demonstrates a complete linear regression workflow: model estimation, statistical inference (confidence & prediction intervals), residual diagnostics, and formal assumption testing.

---

## Overview

| Component                    | Details                                      |
|-----------------------------|----------------------------------------------|
| **Model**                   | `mpg ~ hp` (with intercept)                  |
| **Dataset**                 | `mtcars` (32 observations)                   |
| **Inference**               | 95% Confidence Interval & Prediction Interval |
| **Diagnostics**             | Residuals vs Fitted, Q-Q plot, Scale-Location, Cook’s Distance |
| **Assumption Tests**        | Shapiro-Wilk, Breusch-Pagan, Durbin-Watson   |

---

## Project Structure
```
Regression_Analysis/          (or Horsepower_mpg_Regression)
│
├── .gitignore
├── LICENSE
├── Regression_Analysis.ipynb
├── regression_analysis.py
└── requirements.txt
```

---

## Key Features

- Clean OLS model fitting with `statsmodels`
- Visualisation of the regression line with **confidence** and **prediction** bands
- Comprehensive residual diagnostics (4 classic plots)
- Formal statistical tests for:
  - Normality of residuals
  - Homoscedasticity
  - Autocorrelation
- Influence analysis using Cook’s Distance
- Fully documented and reproducible

---

## Installation

```bash
git clone https://github.com/ewangila/Regression_Analysis.git
cd Regression_Analysis
pip install -r requirements.txt
```
## Usage

### Option 1 — Jupyter Notebook
Open and run `Regression_Analysis.ipynb` cell by cell for the full interactive walkthrough.

### Option 2 — Python Script
```python
from regression_analysis import analyze_mpg_vs_hp

model = analyze_mpg_vs_hp(alpha=0.05)
print(model.summary())
```
## Requirements

- Python ≥ 3.9
- pandas
- numpy
- statsmodels
- matplotlib
- scipy

See `requirements.txt` for recommended versions.

## Author

**Eugin Wangila**  
Data Science & Statistical Modelling

## License
MIT License
