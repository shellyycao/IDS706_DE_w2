import pandas as pd

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

######        

