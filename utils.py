import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from IPython.display import display, HTML
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from matplotlib.table import Table
from scipy.stats import boxcox, yeojohnson
from IPython.display import display
from datetime import datetime

import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# This function prints the number of unique values and the top 15 unique values for each column in a DataFrame.
# It is useful for getting a quick overview of the variety in each column.
def print_unique_values_with_counts(df, cols=None):
    # Check if specific columns are provided, else use all columns from the dataframe
    check_cols = cols if cols is not None else df.columns

    for column in check_cols:
        # Get unique values and their counts
        value_counts = df[column].value_counts()
        num_unique_values = len(value_counts)

        # Print the total number of unique values
        print(f"Column '{column}' has {num_unique_values} unique values.")

        # Print the top 10 unique values with their counts
        top_15 = value_counts.head(15)  # Get the top 15 values and their counts
        print("Top 15 unique values and their counts:")
        for value, count in top_15.items():
            print(f"{value}: {count}")

        print("\n")  # New line for better readability between columns


def model_analysis(X, y, drop=None, predictors=None):

    if 'const' in X.columns:
        X.drop([drop], axis=1, inplace=True)

    # Drop the specified predictor
    if drop is not None and drop in X.columns:
        X_modified = X.drop([drop], axis=1)
    else:
        X_modified = X.copy()

    # If a list of predictors is specified, select only those predictors
    if predictors is not None:
        X_modified = X_modified[predictors]

    # Add a constant to the predictor variable set
    X_with_constant = sm.add_constant(X_modified)

    # Fit the linear model
    lm = sm.OLS(y, X_with_constant).fit()

    # Print the summary of the linear model
    print(lm.summary())

    vif_df = vif_analysis(X_modified, y)
    display(vif_df)

    return X_modified, lm

def vif_analysis(X, y):
    # Calculate VIF
    vif_df = pd.DataFrame()
    vif_df['Features'] = X.columns
    vif_df['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_df['VIF'] = round(vif_df['VIF'], 2)
    vif_df = vif_df.sort_values(by="VIF", ascending=False)
    return vif_df

def analyze_and_transform_feature(df, feature, transformation_type='log'):

    print("Starting analysis and transformation for feature:", feature)

    # Step 1: Plotting original data
    print("Plotting original data...")
    fig, ax = plt.subplots(1, 3, figsize=(15, 3))
    sns.histplot(df[feature], kde=True, ax=ax[0])
    ax[0].set_title('Histogram of ' + feature)
    sns.boxplot(x=df[feature], ax=ax[1])
    ax[1].set_title('Boxplot of ' + feature)
    sns.violinplot(x=df[feature], ax=ax[2])
    ax[2].set_title('Violin Plot of ' + feature)
    plt.tight_layout()
    plt.show()

    # Step 2: Transformation
    print(f"Performing {transformation_type} transformation...")
    df_transformed = df.copy()
    if transformation_type == 'log':
        df_transformed[feature + "_log"] = np.log1p(df_transformed[feature])
        transformed_feature = feature + "_log"
    elif transformation_type == 'z-score':
        df_transformed[feature + "_zscore"] = (df_transformed[feature] - df_transformed[feature].mean()) / df_transformed[feature].std()
        transformed_feature = feature + "_zscore"
    elif transformation_type == 'box-cox':
        # Box-Cox transformation requires positive data
        if df[feature].min() <= 0:
            print(f"Box-Cox transformation not possible for '{feature}' as it contains non-positive values.")
            return df
        df_transformed[feature + "_boxcox"], _ = boxcox(df_transformed[feature])
        transformed_feature = feature + "_boxcox"
    elif transformation_type == 'yeo-johnson':
        df_transformed[feature + "_yeojohnson"], _ = yeojohnson(df_transformed[feature])
        transformed_feature = feature + "_yeojohnson"
    else:
        raise ValueError("Invalid transformation type. Choose 'log', 'z-score', 'box-cox', or 'yeo-johnson'.")

    # Step 3: Plotting transformed data
    print(f"Plotting {transformation_type} transformed data...")
    fig, ax = plt.subplots(1, 3, figsize=(15, 3))
    sns.histplot(df_transformed[transformed_feature], kde=True, ax=ax[0])
    ax[0].set_title(f'Histogram of {transformed_feature}')
    sns.boxplot(x=df_transformed[transformed_feature], ax=ax[1])
    ax[1].set_title(f'Boxplot of {transformed_feature}')
    sns.violinplot(x=df_transformed[transformed_feature], ax=ax[2])
    ax[2].set_title(f'Violin Plot of {transformed_feature}')
    plt.tight_layout()
    plt.show()

    print(f"Analysis and {transformation_type} transformation complete for feature: {feature}")

    return df_transformed

#
# Function to run bivariate analysis of categorical variables versus target
#
def bivariate_cat_analysis(df, cat_vars, target):
    fig, axes = plt.subplots(len(cat_vars), 1, figsize=(15, len(cat_vars) * 6))

    for i, var in enumerate(cat_vars):
        sns.boxplot(data=df, x=var, y=target, ax=axes[i], palette='Set2')
        axes[i].set_title(f'Box Plot of {var} vs cnt')
        axes[i].set_ylabel('Count of Total Rentals')

    plt.tight_layout()
    plt.show()

#   
# Function to run bivariate analysis of continuous variables versus target
#
def bivariate_cont_analysis(df, cont_vars, target):
    fig, axes = plt.subplots(len(cont_vars), 1, figsize=(15, len(cont_vars) * 4))

    for i, var in enumerate(cont_vars):
        # Use regplot with customized scatter and line colors
        sns.regplot(data=df, x=var, y=target, ax=axes[i], 
                    scatter_kws={'color': 'blue', 'alpha': 0.5},  # Blue dots with semi-transparency
                    line_kws={'color': 'red'})  # Red regression line
        axes[i].set_title(f'Scatter Plot of {var} vs cnt with Regression Line')
        axes[i].set_ylabel('Count of Total Rentals')

    plt.tight_layout()
    plt.show()

#
# wrapper over bivariate_cat_analysis and bivariate_cont_analysis, that chooses either of the functions based on the context
#
def bivariate_analysis(df, vars, target):
    
    # Distinguish between categorical and continuous variables
    continuous_vars = []
    categorical_vars = []

    for var in vars:
        # A simple heuristic to distinguish types: by the number of unique values
        # This threshold can be adjusted according to your specific dataset
        if len(df[var].unique()) > 15:  # Assuming a variable is continuous if it has more than 15 unique values
            continuous_vars.append(var)
        else:
            categorical_vars.append(var)

    # Call analysis functions for continuous and categorical variables
    if continuous_vars:
        bivariate_cont_analysis(df, continuous_vars, target)
    
    if categorical_vars:
        bivariate_cat_analysis(df, categorical_vars, target)

#
# Function to run univariate analysis on the continuous variables
#
def univariate_cont_analysis(df, cont_vars):

    # Define the number of subplots based on the number of continuous variables
    n_cols = 2
    n_rows = int(len(cont_vars) / n_cols) + (len(cont_vars) % n_cols > 0)

    # Plot histograms and boxplots for continuous variables
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(14, n_rows * 4))

    for i, var in enumerate(cont_vars):
        # Histogram
        sns.histplot(df[var], kde=True, ax=axes[i // n_cols, (i % n_cols) * 2], color='skyblue')
        axes[i // n_cols, (i % n_cols) * 2].set_title(f'Histogram of {var}')
        
        # Boxplot
        sns.boxplot(x=df[var], ax=axes[i // n_cols, (i % n_cols) * 2 + 1], color='lightgreen')
        axes[i // n_cols, (i % n_cols) * 2 + 1].set_title(f'Boxplot of {var}')

    plt.tight_layout()
    plt.show()

#
# Function to run univariate analysis on the categorical variables
#
def univariate_cat_analysis(df, cat_vars):

    fig, axes = plt.subplots(len(cat_vars), 1, figsize=(8, len(cat_vars) * 4))

    for i, var in enumerate(cat_vars):
        sns.countplot(data=df, y=var, ax=axes[i], order = df[var].value_counts().index, palette='Set3')
        axes[i].set_title(f'Bar Plot of {var}')
        axes[i].set_xlabel('Frequency')

    plt.tight_layout()
    plt.show()

#
# wrapper over univariate_cat_analysis and univariate_cont_analysis, that chooses either of the functions based on the context
#
def univariate_analysis(df, vars):
    
    # Distinguish between categorical and continuous variables
    continuous_vars = []
    categorical_vars = []

    for var in vars:
        # A simple heuristic to distinguish types: by the number of unique values
        # This threshold can be adjusted according to your specific dataset
        if len(df[var].unique()) > 15:  # Assuming a variable is continuous if it has more than 10 unique values
            continuous_vars.append(var)
        else:
            categorical_vars.append(var)

    # Call analysis functions for continuous and categorical variables
    if continuous_vars:
        univariate_cont_analysis(df, continuous_vars)
    
    if categorical_vars:
        univariate_cat_analysis(df, categorical_vars)

#
# Function to print the RFE status
#
def print_rfe_stats(X_train, rfe):
    # Combine the feature names with their respective RFE support flags and rankings
    features = list(zip(X_train.columns, rfe.support_, rfe.ranking_))

    # Sort the features: first by support (True before False), then by ranking
    # Use `~` for negating boolean values instead of `-`
    sorted_features = sorted(features, key=lambda x: (~x[1], x[2]))

    # Display the sorted list
    for feature in sorted_features:
        print(feature)