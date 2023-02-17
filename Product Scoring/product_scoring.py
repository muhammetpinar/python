

import pandas as pd

data = pd.read_csv("product_scoring.csv")

# 1. approach

data["CTR"] = data["clicks"] / data["impressions"]
data.fillna(0,inplace =True)

corr_CTR = data["CTR"].corr(data["clicks"])
corr_CPC = data["cpc"].corr(data["clicks"])

sum_corr = abs(corr_CTR) + abs(corr_CPC) 

weight_CTR = abs(corr_CTR) / sum_corr
weight_CPC = abs(corr_CPC) / sum_corr

data["score"] = (weight_CTR * data["CTR"]) + (weight_CPC * data["cpc"]) 

sorted_data = data.sort_values(by="score", ascending=False)
print(sorted_data.head(10))

data["score2"] = (data["clicks"] * data["cpc"]) / data["impressions"]

sorted_da = data.sort_values("score2", ascending=False)

print(sorted_da.head(10))

# 3. approach

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


X_train, X_test, y_train, y_test = train_test_split(
    data[[ "clicks", "impressions", "cpc"]],
    data["clicks"] > 0, 
    test_size=0.2,
    random_state=42
)


model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

data["score3"] = model.predict_proba(data[["clicks", "impressions", "cpc"]])[:, 1]

sorted_data = data.sort_values("score", ascending=False)

print(sorted_data.head(10))

y_pred = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)

print("AUC on test data: {:.3f}".format(auc))

#4

from scipy.stats import beta

alpha_prior = 1  # Prior parameter for alpha
beta_prior = 1  # Prior parameter for beta

data["alpha_post"] = alpha_prior + data["CTR"]
data["beta_post"] = beta_prior + data["impressions"] - data["CTR"]

data["score4_click_post"] = data.apply(lambda row: beta.mean(row["alpha_post"], row["beta_post"]), axis=1)

# Sort the products by the posterior mean of the true CTR
sorted_data = data.sort_values("score4_click_post", ascending=False)

# Print the top 10 products by posterior mean of the true CTR
print(sorted_data.head(10))


data['general_score'] = (data['score'] + data['score2'] + data['score3'] + data['score4_click_post'])  
sorted_data = data.sort_values("general_score", ascending=False)
