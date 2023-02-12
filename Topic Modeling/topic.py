import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")
tags = pd.read_csv("Tags.csv")
sample_sub = pd.read_csv("SampleSubmission.csv")

print(train.isna().sum)

print(test.isna().sum)


train["Number of Characters"] = train["ABSTRACT"].apply(lambda x: len(str(x)))
test["Number of Characters"] = test["ABSTRACT"].apply(lambda x: len(str(x)))
fig = make_subplots(rows=1, cols=2)
trace1 = go.Histogram(x = train["Number of Characters"])
fig.add_trace(trace1, row=1, col=1)

trace2 = go.Box(y = train["Number of Characters"])
fig.add_trace(trace2, row=1, col=2)
fig.update_layout(showlegend=False)
fig.show()

fig = make_subplots(rows=1, cols=2)
trace1 = go.Histogram(x = test["Number of Characters"])
fig.add_trace(trace1, row=1, col=1)

trace2 = go.Box(y = test["Number of Characters"])
fig.add_trace(trace2, row=1, col=2)
fig.update_layout(showlegend=False)
fig.show()

train['Number of Words'] = train['ABSTRACT'].apply(lambda x: len(str(x).split()))
test['Number of Words'] = test['ABSTRACT'].apply(lambda x: len(str(x).split()))
fig = make_subplots(rows = 1, cols = 2)
trace1 = go.Histogram(x = train['Number of Words'])
fig.add_trace(trace1, row = 1, col = 1)

trace2 = go.Box(y = train['Number of Words'])
fig.add_trace(trace2, row = 1, col = 2)

fig.update_layout(showlegend = False)
fig.show()

fig = make_subplots(rows = 1, cols = 2)
trace1 = go.Histogram(x = test['Number of Words'])
fig.add_trace(trace1, row = 1, col = 1)

trace2 = go.Box(y = test['Number of Words'])
fig.add_trace(trace2, row = 1, col = 2)

fig.update_layout(showlegend = False)
fig.show()

main_tags = ['Computer Science',
 'Mathematics',
 'Physics',
 'Statistics']

countTagsTrain = pd.DataFrame(train[main_tags].sum(axis = 0) / len(train))
countTagsTest = pd.DataFrame(test[main_tags].sum(axis = 0) / len(test))

trace0 = go.Bar(x = countTagsTrain.index, y = countTagsTrain[0],name = 'Train Set')
trace1 = go.Bar(x = countTagsTest.index, y = countTagsTest[0],name = 'Test Set')

fig = go.Figure([trace0,trace1])
fig.show()