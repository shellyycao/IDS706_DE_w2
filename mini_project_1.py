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

pd.set_option("mode.copy_on_write", True)

# Import data
file = pd.read_csv("Ecommerce_Consumer_Behavior_Analysis_Data.csv")
Eco_Con_Beh = pd.DataFrame(file)

# Print first 10 rows of the dataframe
print(Eco_Con_Beh.head(10))

# Inspect the data
print(Eco_Con_Beh.dtypes)
print(Eco_Con_Beh.info())
print(Eco_Con_Beh.describe())

# Check for missing values
print(Eco_Con_Beh.isnull().sum())
# Drop missing values
Eco_Con_Beh = Eco_Con_Beh.dropna()
# Check if still have missing values
print(Eco_Con_Beh.isnull().sum())

# Drop duplicates
Eco_Con_Beh = Eco_Con_Beh.drop_duplicates()

# Save cleaned data to a new CSV file
Eco_Con_Beh.to_csv("Cleaned_Data.csv", index=False)

###############################################
# Data Analysis
###############################################

# 1. Age and gender distribution of customers

# See the age distribution of customers
average_age = Eco_Con_Beh["Age"].mean()
median_age = Eco_Con_Beh["Age"].median()
mode_age = Eco_Con_Beh["Age"].mode()[0]

print(f"The average age of customers is: {average_age}")
print(f"The median age of customers is: {median_age}")
print(f"The mode age of customers is: {mode_age}")

# See the gender distribution of customers
gender_counts = Eco_Con_Beh["Gender"].value_counts()
print("Gender distribution of customers:")
print(gender_counts)

# Plot age distribution with gender differentiation
age_bins = [0, 20, 25, 30, 35, 40, 45, Eco_Con_Beh["Age"].max() + 1]
age_labels = ["<20", "20-25", "25-30", "30-35", "35-40", "40-45", ">45"]
age_groups = pd.cut(
    Eco_Con_Beh["Age"], bins=age_bins, labels=age_labels, right=False)

Eco_Con_Beh = Eco_Con_Beh.copy()
Eco_Con_Beh["Age_Group"] = age_groups  # Create a new column for age groups

# Group by Age_Group and Gender
age_gender_distribution = (
    Eco_Con_Beh.groupby(["Age_Group", "Gender"]).size().unstack(fill_value=0)
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
top_10_categories = Eco_Con_Beh[
    "Purchase_Category"].value_counts().head(10).index
category_age_group = (
    Eco_Con_Beh[Eco_Con_Beh["Purchase_Category"].isin(
        top_10_categories)]
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
    Eco_Con_Beh[Eco_Con_Beh["Purchase_Category"].isin(top_10_categories)]
    .groupby(["Purchase_Category", "Gender"])
    .size()
    .unstack(fill_value=0)
)

cat_gender = cat_gender.loc[
    cat_gender.sum(axis=1).sort_values(ascending=True).index]
cat_gender_pct = cat_gender.div(cat_gender.sum(axis=1), axis=0).fillna(0) * 100

ax = cat_gender.plot(kind="barh", stacked=True, figsize=(9, 6))
ax.set_title("Top 10 Product Categories by Gender (Counts)")
ax.set_xlabel("Count")
ax.set_ylabel("Product Category")

THRESHOLD = 60  # percent


# helper: put % in the middle of a rect
def label_pct(rect, pct):
    x = rect.get_x() + rect.get_width() / 2
    y = rect.get_y() + rect.get_height() / 2
    ax.text(
        x, y,
        f"{pct:.0f}%", ha="center", va="center", fontsize=9, weight="bold")


# highlight segments ≥ THRESHOLD and fade the rest
for col_idx, container in enumerate(ax.containers):
    for row_idx, rect in enumerate(container.patches):
        pct = float(cat_gender_pct.iloc[row_idx, col_idx])
        if pct >= THRESHOLD:
            rect.set_alpha(1.0)
            rect.set_linewidth(1.2)
            rect.set_edgecolor("black")
            label_pct(rect, pct)
        else:
            rect.set_alpha(0.35)

plt.tight_layout()
plt.show()

# 3. Average purchase amount by age group
# Ensure 'Purchase_Amount' is numeric
Eco_Con_Beh[
    "Purchase_Amount_Numeric"] = Eco_Con_Beh["Purchase_Amount"].replace(
    r"[\$,]", "", regex=True
)
Eco_Con_Beh["Purchase_Amount_Numeric"] = pd.to_numeric(
    Eco_Con_Beh["Purchase_Amount_Numeric"], errors="coerce"
)

avg_purchase_by_age_group = Eco_Con_Beh.groupby("Age_Group")[
    "Purchase_Amount_Numeric"
].mean()
print(avg_purchase_by_age_group)

# 4. Relationship between Age Group and Amount Spent
age_purchase = Eco_Con_Beh.groupby("Age")["Purchase_Amount_Numeric"].mean()
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
X = Eco_Con_Beh[["Age"]]
y = Eco_Con_Beh["Purchase_Amount_Numeric"]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

# Not statistically significant!

# I wonder if gender has a significant effect on the amount spent
# by customers.
anova_model1 = smf.ols(
    "Purchase_Amount_Numeric ~ C(Gender)", data=Eco_Con_Beh).fit()
anova_table1 = sm.stats.anova_lm(anova_model1)
print(anova_table1)
# In general, it seems gender does not have a significant effect
# on the amount spent by customers.

# BUT, if we only look at Female and Male (drop Other):
male = Eco_Con_Beh[
    Eco_Con_Beh["Gender"] == "Male"]["Purchase_Amount_Numeric"]
female = Eco_Con_Beh[
    Eco_Con_Beh["Gender"] == "Female"]["Purchase_Amount_Numeric"]
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
    "Purchase_Amount_Numeric ~ C(Income_Level)", data=Eco_Con_Beh
).fit()
anova_table2 = sm.stats.anova_lm(anova_model2)
print(anova_table2)
# Surprisingly, income level does not have a significant effect

# I wonder if time spend on research has a significant effect on
# customer satisfaction
x = Eco_Con_Beh["Time_Spent_on_Product_Research(hours)"]
y = Eco_Con_Beh["Customer_Satisfaction"]
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
Eco_Con_Beh["Gender_Code"] = LabelEncoder().fit_transform(
    Eco_Con_Beh["Gender"])
Eco_Con_Beh["Income_Level_Code"] = LabelEncoder().fit_transform(
    Eco_Con_Beh["Income_Level"]
)
X = Eco_Con_Beh[["Age", "Gender_Code", "Income_Level_Code"]]
y = Eco_Con_Beh["Purchase_Amount_Numeric"]
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
Eco_Con_Beh.to_csv("Processed_Data.csv", index=False)
