###############################################
# Mini Project 1: Data Cleaning and Analysis of
# E-commerce Consumer Behavior
###############################################

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# Import data
file = pd.read_csv("Ecommerce_Consumer_Behavior_Analysis_Data.csv")
df = pd.DataFrame(file)

# Print first 10 rows of the dataframe
print(df.head(10))

# Inspect the data
print(df.dtypes)
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())
# Drop missing values
df = df.dropna()
# Check if still have missing values
print(df.isnull().sum())

# Drop duplicates
df = df.drop_duplicates()
# Check if still have duplicates
print(f"Number of duplicates after dropping: {df.duplicated().sum()}")

# Save cleaned data to a new CSV file
df.to_csv("Cleaned_Data.csv", index=False)

###############################################
# Data Analysis
###############################################

# 1. Age and gender distribution of customers

# See the age distribution of customers
average_age = df["Age"].mean()
median_age = df["Age"].median()
mode_age = df["Age"].mode()[0]

print(f"The average age of customers is: {average_age}")
print(f"The median age of customers is: {median_age}")
print(f"The mode age of customers is: {mode_age}")

# See the gender distribution of customers
gender_counts = df["Gender"].value_counts()
print("Gender distribution of customers:")
print(gender_counts)

# Plot age distribution with gender differentiation
age_bins = [0, 20, 25, 30, 35, 40, 45, df["Age"].max() + 1]
age_labels = ["<20", "20-25", "25-30", "30-35", "35-40", "40-45", ">45"]
age_groups = pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=False)

df = df.copy()
df["Age_Group"] = age_groups  # Create a new column for age groups

# Group by Age_Group and Gender
age_gender_distribution = (
    df.groupby(["Age_Group", "Gender"]).size().unstack(fill_value=0)
)

# Plot bar chart
ax = age_gender_distribution.plot(kind="bar", stacked=True)
ax.set_title("Age Distribution of Customers by Gender")
ax.set_xlabel("Age Group")
ax.set_ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 2. Most popular product categories
top_10_categories = df["Purchase_Category"].value_counts().head(10).index
category_age_group = (
    df[df["Purchase_Category"].isin(top_10_categories)]
    .groupby(["Purchase_Category", "Age_Group"])
    .size()
    .unstack(fill_value=0)
)

# Sort categories by total count for better visualization
category_age_group = category_age_group.loc[
    category_age_group.sum(axis=1).sort_values(ascending=True).index
]

# Plot stacked bar chart for top 10 product categories by age group
ax = category_age_group.plot(kind="barh", stacked=True, colormap="tab10")
ax.set_title("Top 10 Product Categories by Age Group")
ax.set_xlabel("Count")
ax.set_ylabel("Product Category")
plt.tight_layout()
plt.show()

# Plot stacked bar chart for top 10 product categories by gender
cat_gender = (
    df[df["Purchase_Category"].isin(top_10_categories)]
    .groupby(["Purchase_Category", "Gender"])
    .size()
    .unstack(fill_value=0)
)

cat_gender = cat_gender.loc[
    cat_gender.sum(axis=1).sort_values(ascending=True).index]

ax = cat_gender.plot(kind="barh", stacked=True, figsize=(8, 6))
ax.set_title("Top 10 Product Categories by Gender (Counts)")
ax.set_xlabel("Count")
ax.set_ylabel("Product Category")
plt.tight_layout()
plt.show()

# 3. Average purchase amount by age group
# Ensure 'Purchase_Amount' is numeric
df["Purchase_Amount_Numeric"] = df[
    "Purchase_Amount"].replace(r"[\$,]", "", regex=True)
df["Purchase_Amount_Numeric"] = pd.to_numeric(
    df["Purchase_Amount_Numeric"], errors="coerce"
)

avg_purchase_by_age_group = df.groupby("Age_Group")[
    "Purchase_Amount_Numeric"].mean()
print(avg_purchase_by_age_group)

# 4. Relationship between Age Group and Amount Spent
age_purchase = df.groupby("Age")["Purchase_Amount_Numeric"].mean()
z = np.polyfit(age_purchase.index, age_purchase.values, 1)  # Linear fit
p = np.poly1d(z)

plt.figure(figsize=(8, 5))
plt.plot(age_purchase.index, age_purchase.values, "o", color="skyblue")
plt.title("Average Amount Spent by Age (Dot Chart)")
plt.xlabel("Age")
plt.ylabel("Average Purchase Amount")
plt.grid(True, linestyle="--", alpha=0.5)
plt.plot(age_purchase.index, p(age_purchase.index), "r--", label="Trend Line")
plt.tight_layout()
plt.show()

################
# Second, I want to run some statistical tests to some
# of my hypotheses.
################

# I wonder if age has a significant effect on the amount spent
# by customers.
X = df[["Age"]]
y = df["Purchase_Amount_Numeric"]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

# Not statistically significant!

# I wonder if gender has a significant effect on the amount spent
# by customers.
anova_model1 = smf.ols(
    "Purchase_Amount_Numeric ~ C(Gender)", data=df).fit()
anova_table1 = sm.stats.anova_lm(anova_model1)
print(anova_table1)
# In general, it seems gender does not have a significant effect
# on the amount spent by customers.

# BUT, if we only look at Female and Male (drop Other):
male = df[df["Gender"] == "Male"]["Purchase_Amount_Numeric"]
female = df[df["Gender"] == "Female"]["Purchase_Amount_Numeric"]
print("Male mean:", male.mean(), "SD:", male.std())
print("Female mean:", female.mean(), "SD:", female.std())
# Conduct t-test:
t_stat, p_value = stats.ttest_ind(
    male, female, equal_var=False)  # Welch's t-test
print("t-statistic:", t_stat)
print("p-value:", p_value)
# Here, we see a statistically significant difference (if set P at 0.1),
# the negative t-value indicate that males spent less than females
# on average.

# I wonder if income level has a significant effect on the amount
# spent by customers.
anova_model2 = smf.ols(
    "Purchase_Amount_Numeric ~ C(Income_Level)", data=df).fit()
anova_table2 = sm.stats.anova_lm(anova_model2)
print(anova_table2)
# Surprisingly, income level does not have a significant effect

# I wonder if time spend on research has a significant effect on
# customer satisfaction
x = df["Time_Spent_on_Product_Research(hours)"]
y = df["Customer_Satisfaction"]
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
print(model.summary())
# Not statistically significant!

#############################
# Machine Learning Algorithm
#############################

# Note: This code is generated with the help of ChatGPT

# I want to predict the purchase amount based on Gender, Age,
# and Income Level
df["Gender_Code"] = LabelEncoder().fit_transform(df["Gender"])
df["Income_Level_Code"] = LabelEncoder().fit_transform(df["Income_Level"])
X = df[["Age", "Gender_Code", "Income_Level_Code"]]
y = df["Purchase_Amount_Numeric"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=516
)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Coefficients (aligned with X.columns):")
print(pd.Series(model.coef_, index=X.columns))
print("Intercept:", model.intercept_)
print("R²:", r2_score(y_test, y_pred))
print("RMSE:", rmse)
# Unfortunately, the model does not perform well, which indicates
# that these features alone are not sufficient to predict purchase
# amount accurately.


# Save processed data to a new CSV file
df.to_csv("Processed_Data.csv", index=False)
