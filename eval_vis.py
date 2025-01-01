import pandas as pd

# Load the CSV file to inspect the contents
file_path = './evaluation_results_t5.csv'
df = pd.read_csv(file_path)

# Display the first few rows and summary of the dataframe to understand its structure
df.head(), df.describe()
