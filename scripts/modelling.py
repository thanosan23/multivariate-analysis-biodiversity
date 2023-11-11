from utils.read_dataset import read_dataset

import shap

from sklearn.linear_model import LinearRegression

df = read_dataset(data={
        'biodiversity': ['Year', 'number of species'],
        'carbon_dioxide': ['Year', 'Annual'],
        'pollution': ['Year', 'Volatile Organic Compounds'],
}, aggregate_col='Year')

df['Year'] = df['Year'].astype(int)
X_train = df[['Year', "Volatile organic compounds"]]
y_train = df['Annual']

model = LinearRegression()
model.fit(X_train, y_train)

explainer = shap.LinearExplainer(model, X_train)

values = explainer.shap_values(X_train)
# shap.plots.bar(explainer(X_train))
shap.partial_dependence_plot("Year", model.predict, X_train, model_expected_value=True, feature_expected_value=True)
# print(model.score(X_train, y_train))
# print(model.coef_)
# print(model.intercept_)
