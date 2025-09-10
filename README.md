# IDS706_DE_w2

The data is download from Kaggle, link here: 
https://www.kaggle.com/datasets/salahuddinahmedshuvo/ecommerce-consumer-behavior-analysis-data

**The python code is created with the help of Copilot and ChatGPT**

## What is this data?

The E-commerce Consumer Behavior Analysis dataset is a structured collection of demographic and behavioral information about online shoppers. It includes customer details such as age, gender, and income level, along with purchasing factors like product category, purchase amount, frequency, brand loyalty, and satisfaction.

## What did I do?

1. I imported the e-commerce dataset, cleaned it by dropping missing values and duplicates, and saved a cleaned copy for analysis(just in case).
2. I took a look at the demographics and behavior (age/gender distributions, top product categories, and spending by age group), and visualized trends with bar charts and dot plots.
3. I ran some stat test to test the hypotheses about the effect of age, gender, income, and research time on spending and satisfaction using OLS regression, ANOVA, and t-tests.
4. I built a linear regression model with age, gender, and income level as predictors of purchase amount, and evaluated it with R² and RMSE, finding that these features alone don’t explain spending well.

##### Note: I also use ChatGPT to help me convert the code so it can go through "make lint"

## What are some findings(so far)?

1. Customers are spread fairly across age groups, with the largest concentration in the 25–30 range; male and female customers make up the majority, while other gender identities are represented in much smaller proportions.
![Age Distribution by Gender](Age_gender.png)

2. Electronics and Furniture are the most popular categories,customers across all age groups show interest in similar categories. Male customers purchase the most in Health Supplements, while female customers purchase the most in Jewelry & Accessories
![Top 10 Product Categories by Age Group](Cat_Age.png)
![Top 10 Product Categories by Gender](Cat_Gender.png)

3. Income Level was surprisingly not a significant predictor of spending!

4. A linear regression model using Age, Gender, and Income Level to predict purchase amount performed poorly. More informative featuresmay be needed for better predictive performance. **Future investigation need to be done**


-------------------------


## Set up the file

First, you might want to clone the repository in your local enviroment by calling "git clone " in the command line.

In the local environment(VS code), set up the following in the terminal:
- Type in
<pre markdown="1"> ```
make install
``` </pre>
- Run mini_project_1.py
- - You could run it in the interactive window for better results.
