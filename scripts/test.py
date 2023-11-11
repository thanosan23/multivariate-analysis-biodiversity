from utils.read_dataset import read_dataset


df = read_dataset(data={
    'biodiversity': ['Year', 'number of species'],
    'pollution': ['Year', 'Volatile organic compounds']
}, aggregate_col='Year')

print(df.head())
