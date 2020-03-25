from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

header = ['Gender', 'Age', 'Debt', 'Married', 'BankCustomer', 'EducationLevel', 'Ethnicity', 'YearsEmployed', 'PriorDefault', 'Employed', 'CreditScore', 'DriversLicense', 'Citizen', 'ZipCode', 'Income', 'ApprovalStatus']
card_approvals_data = pd.read_csv('cc_approvals.data', names=header)

card_approvals_data = card_approvals_data.replace('?', np.NaN)

le = LabelEncoder()
scaler = StandardScaler()

for col in card_approvals_data:
    if card_approvals_data[col].dtype == 'object':
        card_approvals_data[col] = card_approvals_data[col].fillna(card_approvals_data[col].value_counts().index[0])
        card_approvals_data[col] = le.fit_transform(card_approvals_data[col])

X = card_approvals_data.drop(columns=['ApprovalStatus', 'ZipCode', 'DriversLicense', 'Married'])
y = card_approvals_data['ApprovalStatus']

X_scalled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scalled, y, test_size=.4, random_state=42,stratify=y)

param_grid_knn = [{'n_neighbors':np.array(range(1,15))}]
knn  = KNeighborsClassifier()

grid_model_knn = GridSearchCV(knn,param_grid_knn,cv = 4)
grid_model_knn.fit(X_train, y_train)

cval_score = cross_val_score(grid_model_knn, X_train, y_train, cv=4)

print("Accuracy score:",grid_model_knn.score(X_test, y_test)*100)
print("Cross validation score:",np.mean(cval_score)*100)