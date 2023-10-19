# ========== USING API for conversion ==========
# import jpype
# import asposecells

# jpype.startJVM()
# from asposecells.api import Workbook, SaveFormat

# Create a Workbook object with Excel file's path
# workbook = Workbook("test_data.xlsx")

# Save XLSX as CSV
# workbook.save("test_data.csv", SaveFormat.CSV)


# ========== ANOTHER WAY TO CONVERT ==========
# importing pandas as pd
import pandas as pd

# Specify the path to the XLSX file
xlsx_file_path = 'test_data.xlsx'

# Load the XLSX file into a pandas DataFrame
df = pd.read_excel(xlsx_file_path)

# Specify the path to save the CSV file
csv_file_path = 'test_data.csv'

# Save the DataFrame as a CSV file
df.to_csv(csv_file_path, index=False)

print(f"XLSX file '{xlsx_file_path}' converted and saved as CSV file '{csv_file_path}'.")
