import pandas as pd

###############################
# Test if CSV file is uploaded correctly
###############################

def test_csv_file_upload():
    name = "Ecommerce_Consumer_Behavior_Analysis_Data.csv"
    df = pd.read_csv(name)
    
    # I want to see if my the file is read and stored (not empty)

    if df.empty:
        print("The CSV file is empty! Try again!")
        assert False # assert false since we don't like that!
    else:
        print("The CSV file is not empty! Good job!")
# This can also be done by: assert not df.empty
    
    # I want to see if my header of the file is loaded
    header = [
        "Age",
        "Gender",
        "Purchase_Category",
        "Purchase_Amount",
        "Income_Level"
    ] # I only used these so far
    for i in header:
        if i not in df.columns:
            print(f"Missing column: {i}") # tells which col is missing
            assert False
        else: print("You got this!")

##############################
# Test if data cleaning is done correctly
##############################

def test_data_cleaning():
    file = pd.read_csv("Cleaned_Data.csv")# Read the cleaned data I stored
    df = pd.DataFrame(file)

    # I want to check if my code drops missing values
    if df.isnull().sum().sum() != 0:
        assert False
        print(
        "There are still missing values in the dataframe!")
    else: 
        print("There are no missing values in the dataframe!")

    # I want see if my code drops duplicates
    if df.duplicated().sum() > 0:
        assert False
        print(  
        "There are still duplicate rows in the dataframe!")
    else: 
        print("There are no duplicate rows in the dataframe!")

    # Check my data types that I converted Purchase_Amount to numeric
    file1 = pd.read_csv("Processed_Data.csv")
    df1 = pd.DataFrame(file1)
    if df1["Purchase_Amount_Numeric"].dtype in [float, int, "float64", "int64"]:
        print("The data type of Purchase_Amount_Numeric is correct!")
    else:
        assert False
        print("The data type of Purchase_Amount_Numeric is incorrect!")
    


##############################
# Test if data analysis is done correctly
##############################  

def test_data_analysis():
    file = pd.read_csv("Cleaned_Data.csv")# Read the cleaned data I stored
    df = pd.DataFrame(file)

    # I want to see if my code calculates the average age correctly
    average_age = df["Age"].mean()
    if average_age < 0 or average_age > 100:
        assert False
        print("The average age seems incorrect!")
    else: 
        print("The average age seems correct!")

    # I want to see if my code calculates the median age correctly
    median_age = df["Age"].median()
    if median_age < 0 or median_age > 100:
        assert False
        print("The median age seems incorrect!")
    else: 
        print("The median age seems correct!")

    # I want to see if my code calculates the mode age correctly
    mode_age = df["Age"].mode()[0]
    if mode_age < 0 or mode_age > 100:
        assert False
        print("The mode age seems incorrect!")
    else: 
        print("The mode age seems correct!")


def test_age_groups():
    df = pd.read_csv("Processed_Data.csv")
    # I want to see if my code calculates the age groups correctly
    if df["Age_Group"].isnull().all():
        print("Age_Group values are missing")
        assert False
    else:
        print("Age_Group values are assigned correctly")
    


