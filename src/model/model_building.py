# import numpy as np
# import pandas as pd
# import pickle
# #
# from sklearn.ensemble import GradientBoostingClassifier

# # fetch the data from data/processed
# train_data = pd.read_csv('./data/features/train_bow.csv')

# X_train = train_data.iloc[:,0:-1].values
# y_train = train_data.iloc[:,-1].values

# # Define and train the XGBoost model

# clf = GradientBoostingClassifier(n_estimators=50)
# clf.fit(X_train, y_train)

# # save
# pickle.dump(clf, open('model.pkl','wb'))


import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
import os

# Choose feature type
FEATURE_TYPE = "tfidf"  # change to "bow" if you want BoW

# Load training features
train_data = pd.read_csv(f'./data/features/train_{FEATURE_TYPE}.csv')

# Split features and labels
X_train = train_data.iloc[:, 0:-1].values
y_train = train_data.iloc[:, -1].values

# Define and train the Gradient Boosting model
clf = GradientBoostingClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train)

# Make sure the model folder exists
os.makedirs("model", exist_ok=True)

# Save the trained model
model_path = os.path.join("model", "model.pkl")
pickle.dump(clf, open(model_path, 'wb'))

print(f"Model trained and saved at {model_path} using {FEATURE_TYPE} features")
