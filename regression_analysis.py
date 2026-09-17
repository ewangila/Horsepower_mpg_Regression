import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats

def analyze_mpg_vs_hp(alpha: float = 0.05) -> sm.regression.linear_model.RegressionResults:
    """
    Loads the mtcars dataset, fits an OLS regression model (mpg ~ hp),
    performs assumption diagnostics, and visualizes results with 95% CI/PI.
    
    Args:
        alpha (float): Significance level for confidence/prediction intervals (default: 0.05 for 95%)
    
    Returns:
        sm.regression.linear_model.RegressionResults: Fitted OLS model object for downstream analysis
    """
    try:
        # Step 1 & 2: Import Libraries and Load Dataset
        mtcars = sm.datasets.get_rdataset("mtcars").data

        # Step 3: Define Variables (Adding constant for the intercept)
        X = sm.add_constant(mtcars['hp'])
        y = mtcars['mpg']

        # Step 4: Fit the Linear Regression Model
        model = sm.OLS(y, X).fit()

        # Step 5 & 6: Calculate Confidence (CI) and Prediction Intervals (PI)
        # Generate a continuous range of HP values for smooth plotting
        x_range = np.linspace(X['hp'].min(), X['hp'].max(), 100)
        X_pred = sm.add_constant(x_range)
        
        # Extract predictions and intervals
        predictions = model.get_prediction(X_pred)
        intervals = predictions.summary_frame(alpha=alpha)

        # Step 7: Visualize the Main Regression Results (with Matplotlib style fallback)
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            try:
                plt.style.use('seaborn-whitegrid')
            except OSError:
                plt.style.use('ggplot')

        fig, ax = plt.subplots(figsize=(10, 6))

        # Raw data points
        ax.scatter(mtcars['hp'], mtcars['mpg'], facecolors='none', 
                   edgecolors='black', label='Observed Data', zorder=5)

        # Regression line
        ax.plot(x_range, intervals['mean'], color='blue', 
                linewidth=2, label='OLS Regression Line')

        # 95% Confidence Interval (for the Mean)
        ax.fill_between(x_range, intervals['mean_ci_lower'], intervals['mean_ci_upper'],
                        color='blue', alpha=0.2, label=f'{(1-alpha)*100:.0f}% Confidence Interval (Mean)')

        # 95% Prediction Interval (for a Single Value)
        ax.fill_between(x_range, intervals['obs_ci_lower'], intervals['obs_ci_upper'],
                        color='red', alpha=0.15, label=f'{(1-alpha)*100:.0f}% Prediction Interval (Single y)')

        # Plot formatting
        ax.set_title('Inference on MPG based on Horsepower (mtcars)', fontsize=14, weight='bold')
        ax.set_xlabel('Horsepower (hp)', fontsize=12)
        ax.set_ylabel('Miles Per Gallon (mpg)', fontsize=12)
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

        # Step 8: Print Complete Model Summary and Interpretation
        print("\nREGRESSION MODEL SUMMARY")
        print(model.summary())
        
        # Step 9: Additional Model Diagnostics
        print("\nMODEL FIT DIAGNOSTICS")
        print(f"R-squared: {model.rsquared:.4f}")
        print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
        print(f"F-statistic: {model.fvalue:.4f}")
        print(f"Prob (F-statistic): {model.f_pvalue:.4e}")
        print(f"AIC: {model.aic:.2f}")
        print(f"BIC: {model.bic:.2f}")
        
        # Step 10: Assumption Diagnostics - Create 2x2 subplot grid
        print("\nCHECKING REGRESSION ASSUMPTIONS")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Regression Diagnostics - Assumption Checking', fontsize=14, weight='bold')
        
        # Get residuals and fitted values
        residuals = model.resid
        fitted_values = model.fittedvalues
        
        # 1. Residuals vs Fitted (checks linearity and homoscedasticity)
        axes[0, 0].scatter(fitted_values, residuals, facecolors='none', edgecolors='black')
        axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted Values\n(Check: horizontal band around 0)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Q-Q Plot (checks normality of residuals)
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Normal Q-Q Plot\n(Check: points follow diagonal line)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Scale-Location Plot (checks homoscedasticity)
        standardized_residuals = residuals / np.std(residuals)
        axes[1, 0].scatter(fitted_values, np.sqrt(np.abs(standardized_residuals)), 
                          facecolors='none', edgecolors='black')
        axes[1, 0].set_xlabel('Fitted Values')
        axes[1, 0].set_ylabel('√|Standardized Residuals|')
        axes[1, 0].set_title('Scale-Location Plot\n(Check: roughly horizontal trend)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Influence Diagnostics (Cook's Distance)
        influence = model.get_influence()
        cooks_d = influence.cooks_distance[0]
        
        axes[1, 1].stem(range(len(cooks_d)), cooks_d, markerfmt=',', basefmt=' ')
        axes[1, 1].axhline(y=4/len(y), color='red', linestyle='--', linewidth=2, label="Threshold (4/n)")
        axes[1, 1].set_xlabel('Observation Index')
        axes[1, 1].set_ylabel("Cook's Distance")
        axes[1, 1].set_title("Cook's Distance\n(Check: most points below threshold)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Step 11: Outlier and Influence Diagnostics Report
        print("\nOutlier & Influence Analysis:")
        
        # Identify high-leverage/influential points
        threshold = 4 / len(y)
        high_influence_idx = np.where(cooks_d > threshold)[0]
        
        if len(high_influence_idx) > 0:
            print(f"Observations with Cook's Distance > {threshold:.4f}:")
            for idx in high_influence_idx:
                print(f"Row {idx}: Cook's D = {cooks_d[idx]:.4f}, HP = {mtcars.iloc[idx]['hp']}, MPG = {mtcars.iloc[idx]['mpg']}")
        else:
            print(f"No observations exceed Cook's Distance threshold of {threshold:.4f}")
        
        # Assumption Test Summary
        print("\nASSUMPTION CHECK SUMMARY:")
        
        # Normality test (Shapiro-Wilk)
        stat, p_value = stats.shapiro(residuals)
        print(f"Normality of Residuals (Shapiro-Wilk Test): p-value = {p_value:.4f}")
        
        # Homoscedasticity test (Breusch-Pagan)
        from statsmodels.stats.diagnostic import het_breuschpagan
        bp_stat, bp_pval, _, _ = het_breuschpagan(residuals, X)
        print(f"Homoscedasticity (Breusch-Pagan Test): p-value = {bp_pval:.4f}")
        
        # Autocorrelation test (Durbin-Watson)
        from statsmodels.stats.stattools import durbin_watson
        dw_stat = durbin_watson(residuals)
        print(f"Autocorrelation (Durbin-Watson): {dw_stat:.4f} (2 = no autocorrelation)")
        
        # Step 12: Return model for downstream use
        return model
        
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        return None

if __name__ == "__main__":
    model = analyze_mpg_vs_hp(alpha=0.05)
    
    if model is not None:
        print("\nModel successfully fitted and returned for further use")
        print(f"Model coefficients accessible via: model.params")
        print(f"Predictions accessible via: model.predict(new_data)")