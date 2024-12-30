import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from prophet import Prophet
from datetime import date, timedelta
from hierarchicalforecast.utils import aggregate
from statsforecast import StatsForecast
from statsforecast.models import HoltWinters

from hierarchicalforecast.methods import BottomUp, TopDown, MinTrace, ERM, OptimalCombination
from hierarchicalforecast.core import HierarchicalReconciliation

from sklearn.metrics import mean_squared_error

import streamlit as st
import altair as alt

st.title('Commodity/Sub-Commodity Forecasting')
# data = pd.read_csv('C:/DummyPath/EXPORT-time-series-forecasting.csv')
# #data.head()

# data['COMM-DESC'] = data['COMM-DESC'].str.replace('/','-')
# data['SUB-COMM-DESC'] = data['SUB-COMM-DESC'].str.replace('/','-')
# df = data[['DATE','COMM-DESC','SUB-COMM-DESC','UNITS']]
# df = df.rename(columns={'DATE':'ds','UNITS':'y'})
# df['ds'] = pd.to_datetime(df['ds']).dt.date

# max_date = pd.to_datetime(df['ds'].max()).date()
# start_val_date = max_date - timedelta(days=28)

# train = df.loc[df['ds'] < start_val_date]
# valid = df.loc[(df['ds'] >= start_val_date) & (df['ds'] < max_date)]
# h = valid['ds'].nunique()

# spec = [['COMM-DESC']
#         ,['COMM-DESC','SUB-COMM-DESC']]

# train_agg, S_train, tags = aggregate(train,spec)
# valid_agg, _, _ = aggregate(valid, spec)

# model = StatsForecast(models=[HoltWinters(season_length=52, error_type='A')], 
#                               freq='W', n_jobs=-1)
# model.fit(train_agg)

# p = model.forecast(h=h, fitted=True)
# p_fitted = model.forecast_fitted_values()

# reconcilers = [BottomUp(), 
#                TopDown(method='forecast_proportions'),
#                TopDown(method='average_proportions'),
#                TopDown(method='proportion_averages'),
#             #    MinTrace(method='ols', nonnegative=True),
#             #    MinTrace(method='wls_struct', nonnegative=True),
#             #    MinTrace(method='wls_var', nonnegative=True),
#             #    MinTrace(method='mint_shrink', nonnegative=True), 
#             #    MinTrace(method='mint_cov', nonnegative=True),
#                OptimalCombination(method='ols', nonnegative=True), 
#                OptimalCombination(method='wls_struct', nonnegative=True),
#                ERM(method='closed'),
#                ERM(method='reg'),
#                ERM(method='reg_bu'),
#               ]

# rec_model = HierarchicalReconciliation(reconcilers=reconcilers)

# p_rec = rec_model.reconcile(Y_hat_df=p, Y_df=p_fitted, S=S_train, tags=tags)

# p_rec['ds'] = pd.to_datetime(p_rec['ds'])
# valid_agg['ds'] = pd.to_datetime(valid_agg['ds'])

# p_rec_ = p_rec.merge(valid_agg, on=['ds', 'unique_id'], how='left')
# p_rec_['y'] = p_rec_['y'].fillna(0)
# p_rec_ = p_rec_.reset_index(drop=False)

# rmse = dict()
# for model_ in p_rec.columns[1:]:
#     rmse_ = mean_squared_error(p_rec_['y'].values, p_rec_[model_].values, squared=False)/1e3
#     # get only the model name
#     model__ = model_.split('/')[-1]
#     rmse[model__] = rmse_
# st.dataframe(pd.DataFrame(rmse, index=['RMSE']).T.sort_values('RMSE'))#.to_excel('HoltWinters-Forecast-RMSE.xlsx')
#pd.DataFrame(rmse, index=['RMSE']).T.sort_values('RMSE')

data = pd.read_excel('C:/DummyPath/HoltWinters-Forecast.xlsx')
rmse = pd.read_excel('C:/DummyPath/HoltWinters-Forecast-RMSE.xlsx')



st.dataframe(rmse)

rec_method = 'HoltWinters/' + rmse.iloc[0,0]
st.write(rec_method)

data = data.rename(columns = {rec_method : 'Forecast','ds': 'DATE','y':'NET SALES'})

chart_data = data.groupby(['DATE']).agg({'NET SALES':'sum','Forecast' :'sum'}).reset_index()
chart_data = chart_data.rename(columns = {rec_method : 'Forecast','ds': 'DATE','y':'NET SALES'})
chart_data['DATE'] = pd.to_datetime(chart_data['DATE']).dt.date
#chart_data = chart_data.rename(columns={'DATE':'index'}).set_index('index')
#st.line_chart(chart_data, y= ['NET SALES','Forecast'])



data_melted = chart_data.melt('DATE', var_name='type', value_name='sales', value_vars=['NET SALES','Forecast'])
line_chart = alt.Chart(data_melted).mark_line(point=True).encode(
    x='DATE:T',
    y='sales:Q',
    color='type:N'
).properties(title='Net Sales vs Forecasted Sales')
st.altair_chart(line_chart, use_container_width = True)



# Get unique categories
categories = data['unique_id'].unique()

# Loop through each category and create a line chart
for category in categories:
    st.write(f"Category: {category}")
    
    # Filter data for the current category
    category_data = data[data['unique_id'] == category]
    
    # Melt the DataFrame to long format
    category_data_melted = category_data.melt('DATE', var_name='type', value_name='sales', value_vars=['NET SALES', 'Forecast'])
    
    # Create an Altair line chart
    line_chart = alt.Chart(category_data_melted).mark_line(point=True).encode(
        x='DATE:T',
        y='sales:Q',
        color='type:N'
    ).properties(
        title=f"Actual vs Forecast Sales for Category {category}"
    )
    
    # Display the chart in Streamlit
    st.altair_chart(line_chart, use_container_width=True)