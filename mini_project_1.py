###############################################
# Mini Project 1: Data Cleaning and Analysis of E-commerce Consumer Behavior

#Import libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import data:
file = pd.read_csv('Ecommerce_Consumer_Behavior_Analysis_Data.csv')
df = pd.DataFrame(file)

#print first 10 rows of the dataframe
print(df.head(10))

#Inspect the data 
print(df.dtypes)
print(df.info())
print(df.describe())

#Check for missing values
print(df.isnull().sum())
#Drop missing values
df = df.dropna()
#Check if still have missing values
print(df.isnull().sum())

#Drop duplicates
df = df.drop_duplicates()
#Check if still have duplicates
print(f"Number of duplicates after dropping: {df.duplicated().sum()}")


#Save cleaned data to a new CSV file
df.to_csv('Cleaned_Data.csv', index=False)

###############################################
###############################################

#Data Analysis

#First, I want to understand the demographics and purchasing behavior of customers.

#1. Age and gender distribution of customers

#See the age distribution of customers
average_age = df['Age'].mean()
median_age = df['Age'].median()
mode_age = df['Age'].mode()[0]
print(f"The average age of customers is: {average_age}")
print(f"The median age of customers is: {median_age}")
print(f"The mode age of customers is: {mode_age}")

#See the gender distribution of customers
gender_counts = df['Gender'].value_counts()
print("Gender distribution of customers:")
print(gender_counts)

#Plot age distribution with gender differentiation:
# Create age groups
age_bins = [0, 20, 25, 30, 35, 40, 45, df['Age'].max() + 1]
age_labels = ['<20', '20-25', '25-30', '30-35', '35-40', '40-45', '>45']
age_groups = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
df = df.copy()
df['Age_Group'] = age_groups # Create a new column for age groups

# Group by Age_Group and Gender
age_gender_distribution = df.groupby(['Age_Group', 'Gender']).size().unstack(fill_value=0)

# Plot bar chart
age_gender_distribution.plot(kind='bar', stacked=True)
plt.title("Age Distribution of Customers by Gender")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


#2. Most popular product categories

# Count occurrences of each product category, get top 10

top_10_categories = df['Purchase_Category'].value_counts().head(10).index
category_age_group = df[df['Purchase_Category'].isin(top_10_categories)].groupby(['Purchase_Category', 'Age_Group']).size().unstack(fill_value=0)

# Sort categories by total count for better visualization
category_age_group = category_age_group.loc[category_age_group.sum(axis=1).sort_values(ascending=True).index]

# Plot stacked bar chart for top 10 product categories by age group
category_age_group.plot(kind='barh', stacked=True, colormap='tab10')
plt.title("Top 10 Product Categories by Age Group")
plt.xlabel("Count")
plt.ylabel("Product Category")
plt.tight_layout()
plt.show()

#3. Average purchase amount by age group

# Ensure 'Purchase_Amount' is numeric (It is object type in the original data)
df['Purchase_Amount_Numeric'] = df['Purchase_Amount'].replace('[\$,]', '', regex=True)
df['Purchase_Amount_Numeric'] = pd.to_numeric(df['Purchase_Amount_Numeric'], errors='coerce')

avg_purchase_by_age_group = df.groupby('Age_Group')['Purchase_Amount_Numeric'].mean()
print("Average Purchase Amount by Age Group:")
print(avg_purchase_by_age_group)

# 4. Relationship between Age Group and Amount Spent

# Plot average amount spent by age as a dot chart
age_purchase = df.groupby('Age')['Purchase_Amount_Numeric'].mean()
z = np.polyfit(age_purchase.index, age_purchase.values, 1)  # 1 = linear fit
p = np.poly1d(z)
plt.figure(figsize=(8, 5))
plt.plot(age_purchase.index, age_purchase.values, 'o', color='skyblue')
plt.title("Average Amount Spent by Age (Dot Chart)")
plt.xlabel("Age")
plt.ylabel("Average Purchase Amount")
plt.grid(True, linestyle='--', alpha=0.5)
plt.plot(age_purchase.index, p(age_purchase.index), "r--", label="Trend Line")
plt.tight_layout()
plt.show()

#################################################
