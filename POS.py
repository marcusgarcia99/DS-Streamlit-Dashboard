import pandas as pd
import numpy as np
import pyodbc
import streamlit as st
import time
import matplotlib.pyplot as plt
from matplotlib import ticker
from numerize import numerize
import altair as alt

import tkinter as tk
from tkinter import filedialog
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout='wide')

pd.set_option("styler.format.thousands",",")

st.sidebar.header('Data Science Dashboard `version 1.1`')

@st.cache_resource
def db_connect():
    server = 'dummy_server' 
    database = 'dummy_database' 
    username = 'dummy_login' 
    password = 'dummy_password' 
    return pyodbc.connect('DRIVER={Dummy Driver};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)

cnxn = db_connect()

@st.cache_data(ttl=6000)
def load_data(query, cols):
    with cnxn.cursor() as cur:
        cur.execute(query)
        a = cur.fetchall()
        return pd.DataFrame.from_records(a,columns=cols)


dates_query = '''
SELECT -- CODE HAS BEEN REMOVED FOR SECURITY REASONS'''

df_date_cols = ['-- CODE HAS BEEN REMOVED FOR SECURITY REASONS']

df_dates = load_data(dates_query,df_date_cols)

item_query = '''SELECT 
    -- CODE HAS BEEN REMOVED FOR SECURITY REASONS'''

df_items_cols = ['-- CODE HAS BEEN REMOVED FOR SECURITY REASONS']

df_items = load_data(item_query, df_items_cols)

store_query = '''
SELECT
-- CODE HAS BEEN REMOVED FOR SECURITY REASONS
'''

df_store_cols = ['-- CODE HAS BEEN REMOVED FOR SECURITY REASONS']

df_store = load_data(store_query, df_store_cols)


if 'ad_weeks' not in st.session_state:
    st.session_state['ad_weeks'] = list(df_dates['AD WEEK'].sort_values().unique())
if 'fiscal_weeks' not in st.session_state:
    st.session_state['fiscal_weeks'] = list(df_dates['FISCAL WEEK'].sort_values().unique())
if 'dates' not in st.session_state:
    st.session_state['dates'] = list(df_dates['TRAN DATE'].sort_values().unique())

def date_slicer_update():

    if st.session_state['date_slicer'] == [] and st.session_state['fiscal_slicer'] == [] and st.session_state['ad_slicer'] == []:
        st.session_state['ad_weeks'] = list(df_dates['AD WEEK'].sort_values().unique())
        st.session_state['fiscal_weeks'] = list(df_dates['FISCAL WEEK'].sort_values().unique())
        st.session_state['dates'] = list(df_dates['TRAN DATE'].sort_values().unique())

    if st.session_state['date_slicer'] != [] and (st.session_state['fiscal_slicer'] != [] or st.session_state['ad_slicer'] != []):
        st.session_state['fiscal_weeks'] = list(df_dates['FISCAL WEEK'][df_dates['TRAN DATE'].isin(st.session_state.date_slicer)].unique())        
        st.session_state['ad_weeks'] = list(df_dates['AD WEEK'][df_dates['TRAN DATE'].isin(st.session_state.date_slicer)].unique())          

    if st.session_state['ad_slicer'] != [] and st.session_state['fiscal_slicer'] != [] and st.session_state['date_slicer'] == []:
        st.session_state['dates'] = list(df_dates.loc[np.where((df_dates['AD WEEK'].isin(st.session_state.ad_slicer) & df_dates['FISCAL WEEK'].isin(st.session_state.fiscal_slicer)))]['TRAN DATE'])

    if st.session_state['ad_slicer'] != [] and st.session_state['fiscal_slicer'] == [] and st.session_state['date_slicer'] == []:
        st.session_state['fiscal_weeks'] = list(df_dates['FISCAL WEEK'][df_dates['AD WEEK'].isin(st.session_state.ad_slicer)].unique())
        st.session_state['dates'] = list(df_dates['TRAN DATE'][df_dates['AD WEEK'].isin(st.session_state.ad_slicer)].unique())
    if st.session_state['fiscal_slicer'] != [] and st.session_state['ad_slicer'] == [] and st.session_state['date_slicer'] == []:
        st.session_state['ad_weeks'] = list(df_dates['AD WEEK'][df_dates['FISCAL WEEK'].isin(st.session_state.fiscal_slicer)].unique())
        st.session_state['dates'] = list(df_dates['TRAN DATE'][df_dates['FISCAL WEEK'].isin(st.session_state.fiscal_slicer)].unique())
    if st.session_state['date_slicer'] != [] and st.session_state['ad_slicer'] == [] and st.session_state['fiscal_slicer'] == []:
        st.session_state['fiscal_weeks'] = list(df_dates['FISCAL WEEK'][df_dates['TRAN DATE'].isin(st.session_state.date_slicer)].unique())
        st.session_state['ad_weeks'] = list(df_dates['AD WEEK'][df_dates['TRAN DATE'].isin(st.session_state.date_slicer)].unique())

    item_groupings = ['-- CODE HAS BEEN REMOVED FOR SECURITY REASONS']
item_groupings = ['-- CODE HAS BEEN REMOVED FOR SECURITY REASONS']



row1_col1, row1_col2, row1_col3 = st.columns(3)

# AD SLICER
with row1_col1:
    ad_wk_input = list(st.multiselect('Select AD Week(s):',st.session_state['ad_weeks'], key = 'ad_slicer', on_change=date_slicer_update))
# FISCAL SLICER
with row1_col2:
    fiscal_wk_input = list(st.multiselect('Select FISCAL Week(s):',st.session_state['fiscal_weeks'], key = 'fiscal_slicer', on_change=date_slicer_update))
with row1_col3:
    date_input = list(st.multiselect('Select DATE(s):', st.session_state['dates'], key = 'date_slicer',on_change=date_slicer_update))  




if 'comms' not in st.session_state:
    st.session_state['comms'] = list(df_items['COMMODITY'].sort_values().unique())
if 'subcomms' not in st.session_state:
    st.session_state['subcomms'] = list(df_items['SUB_COMMODITY'].sort_values().unique())
if 'rpgs' not in st.session_state:
    st.session_state['rpgs'] = list(df_items['RETAIL PRICE GROUP'].sort_values().unique())
if 'items' not in st.session_state:
    st.session_state['items'] = list(df_items['ITEM'].sort_values().unique())

if 'stores' not in st.session_state:
    st.session_state['stores'] = list(df_store['STORE'].sort_values().unique())

if 'cms' not in st.session_state:
    st.session_state['cms'] = list(df_items['CM'].sort_values().unique())
if 'depts' not in st.session_state:
    st.session_state['depts'] = list(df_items['DEPT'].sort_values().unique())
if 'vendors' not in st.session_state:
    st.session_state['vendors'] = list(df_items['VENDOR'].sort_values().unique())

def item_slicer_update():
    if st.session_state['item_slicer']:
        st.session_state['comms'] = list(df_items['COMMODITY'][df_items['ITEM'].isin(st.session_state['item_slicer'])].sort_values().unique())
        st.session_state['subcomms'] = list(df_items['SUB_COMMODITY'][df_items['ITEM'].isin(st.session_state['item_slicer'])].sort_values().unique())
        st.session_state['rpgs'] = list(df_items['RETAIL PRICE GROUP'][df_items['ITEM'].isin(st.session_state['item_slicer'])].sort_values().unique())
        st.session_state['items'] = list(df_items['ITEM'].sort_values().unique())
        st.session_state['cms'] = list(df_items['CM'][df_items['ITEM'].isin(st.session_state['item_slicer'])].sort_values().unique())
        st.session_state['depts'] = list(df_items['DEPT'][df_items['ITEM'].isin(st.session_state['item_slicer'])].sort_values().unique())
        st.session_state['vendors'] = list(df_items['VENDOR'][df_items['ITEM'].isin(st.session_state['item_slicer'])].sort_values().unique())
    elif st.session_state['cm_slicer']:
        st.session_state['comms'] = list(df_items['COMMODITY'][df_items['CM'].isin(st.session_state['cm_slicer'])].sort_values().unique())
        st.session_state['subcomms'] = list(df_items['SUB_COMMODITY'][df_items['CM'].isin(st.session_state['cm_slicer'])].sort_values().unique())
        st.session_state['rpgs'] = list(df_items['RETAIL PRICE GROUP'][df_items['CM'].isin(st.session_state['cm_slicer'])].sort_values().unique())
        st.session_state['items'] = list(df_items['ITEM'][df_items['CM'].isin(st.session_state['cm_slicer'])].sort_values().unique())
        st.session_state['cms'] = list(df_items['CM'].sort_values().unique())
        st.session_state['depts'] = list(df_items['DEPT'][df_items['CM'].isin(st.session_state['cm_slicer'])].sort_values().unique())
        st.session_state['vendors'] = list(df_items['VENDOR'][df_items['CM'].isin(st.session_state['cm_slicer'])].sort_values().unique())
    elif st.session_state['dept_slicer']:
        st.session_state['comms'] = list(df_items['COMMODITY'][df_items['DEPT'].isin(st.session_state['dept_slicer'])].sort_values().unique())
        st.session_state['subcomms'] = list(df_items['SUB_COMMODITY'][df_items['DEPT'].isin(st.session_state['dept_slicer'])].sort_values().unique())
        st.session_state['rpgs'] = list(df_items['RETAIL PRICE GROUP'][df_items['DEPT'].isin(st.session_state['dept_slicer'])].sort_values().unique())
        st.session_state['items'] = list(df_items['ITEM'][df_items['DEPT'].isin(st.session_state['dept_slicer'])].sort_values().unique())
        st.session_state['cms'] = list(df_items['CM'][df_items['DEPT'].isin(st.session_state['dept_slicer'])].sort_values().unique())
        st.session_state['depts'] = list(df_items['DEPT'].sort_values().unique())
        st.session_state['vendors'] = list(df_items['VENDOR'][df_items['DEPT'].isin(st.session_state['dept_slicer'])].sort_values().unique())
    elif st.session_state['comm_slicer']:
        st.session_state['comms'] = list(df_items['COMMODITY'].sort_values().unique())
        st.session_state['subcomms'] = list(df_items['SUB_COMMODITY'][df_items['COMMODITY'].isin(st.session_state['comm_slicer'])].sort_values().unique())
        st.session_state['rpgs'] = list(df_items['RETAIL PRICE GROUP'][df_items['COMMODITY'].isin(st.session_state['comm_slicer'])].sort_values().unique())
        st.session_state['items'] = list(df_items['ITEM'][df_items['COMMODITY'].isin(st.session_state['comm_slicer'])].sort_values().unique())
        st.session_state['cms'] = list(df_items['CM'][df_items['COMMODITY'].isin(st.session_state['comm_slicer'])].sort_values().unique())
        st.session_state['depts'] = list(df_items['DEPT'][df_items['COMMODITY'].isin(st.session_state['comm_slicer'])].sort_values().unique())
        st.session_state['vendors'] = list(df_items['VENDOR'][df_items['COMMODITY'].isin(st.session_state['comm_slicer'])].sort_values().unique())
    elif st.session_state['subcomm_slicer']:
        st.session_state['comms'] = list(df_items['COMMODITY'][df_items['SUB_COMMODITY'].isin(st.session_state['subcomm_slicer'])].sort_values().unique())
        st.session_state['subcomms'] = list(df_items['SUB_COMMODITY'].sort_values().unique())
        st.session_state['rpgs'] = list(df_items['RETAIL PRICE GROUP'][df_items['SUB_COMMODITY'].isin(st.session_state['subcomm_slicer'])].sort_values().unique())
        st.session_state['items'] = list(df_items['ITEM'][df_items['SUB_COMMODITY'].isin(st.session_state['subcomm_slicer'])].sort_values().unique())
        st.session_state['cms'] = list(df_items['CM'][df_items['SUB_COMMODITY'].isin(st.session_state['subcomm_slicer'])].sort_values().unique())
        st.session_state['depts'] = list(df_items['DEPT'][df_items['SUB_COMMODITY'].isin(st.session_state['subcomm_slicer'])].sort_values().unique())
        st.session_state['vendors'] = list(df_items['VENDOR'][df_items['SUB_COMMODITY'].isin(st.session_state['subcomm_slicer'])].sort_values().unique())
    elif st.session_state['rpg_slicer']:
        st.session_state['comms'] = list(df_items['COMMODITY'][df_items['RETAIL PRICE GROUP'].isin(st.session_state['rpg_slicer'])].sort_values().unique())
        st.session_state['subcomms'] = list(df_items['SUB_COMMODITY'][df_items['RETAIL PRICE GROUP'].isin(st.session_state['rpg_slicer'])].sort_values().unique())
        st.session_state['rpgs'] = list(df_items['RETAIL PRICE GROUP'].sort_values().unique())
        st.session_state['items'] = list(df_items['ITEM'][df_items['RETAIL PRICE GROUP'].isin(st.session_state['rpg_slicer'])].sort_values().unique())
        st.session_state['cms'] = list(df_items['CM'][df_items['RETAIL PRICE GROUP'].isin(st.session_state['rpg_slicer'])].sort_values().unique())
        st.session_state['depts'] = list(df_items['DEPT'][df_items['RETAIL PRICE GROUP'].isin(st.session_state['rpg_slicer'])].sort_values().unique())
        st.session_state['vendors'] = list(df_items['VENDOR'][df_items['RETAIL PRICE GROUP'].isin(st.session_state['rpg_slicer'])].sort_values().unique())

    elif st.session_state['vendor_slicer']:
        st.session_state['comms'] = list(df_items['COMMODITY'][df_items['VENDOR'].isin(st.session_state['vendor_slicer'])].sort_values().unique())
        st.session_state['subcomms'] = list(df_items['SUB_COMMODITY'][df_items['VENDOR'].isin(st.session_state['vendor_slicer'])].sort_values().unique())
        st.session_state['rpgs'] = list(df_items['RETAIL PRICE GROUP'][df_items['VENDOR'].isin(st.session_state['vendor_slicer'])].sort_values().unique())
        st.session_state['items'] = list(df_items['ITEM'][df_items['VENDOR'].isin(st.session_state['vendor_slicer'])].sort_values().unique())
        st.session_state['cms'] = list(df_items['CM'][df_items['VENDOR'].isin(st.session_state['vendor_slicer'])].sort_values().unique())
        st.session_state['depts'] = list(df_items['DEPT'][df_items['VENDOR'].isin(st.session_state['vendor_slicer'])].sort_values().unique())
        st.session_state['vendors'] = list(df_items['VENDOR'].sort_values().unique())
    else:
        st.session_state['comms'] = list(df_items['COMMODITY'].sort_values().unique())
        st.session_state['subcomms'] = list(df_items['SUB_COMMODITY'].sort_values().unique())
        st.session_state['rpgs'] = list(df_items['RETAIL PRICE GROUP'].sort_values().unique())
        st.session_state['items'] = list(df_items['ITEM'].sort_values().unique())
        st.session_state['cms'] = list(df_items['CM'].sort_values().unique())
        st.session_state['depts'] = list(df_items['DEPT'].sort_values().unique())
        st.session_state['vendors'] = list(df_items['VENDOR'].sort_values().unique())


row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)

# COMMODITY SLICER (AFFECTED BY RPGs)
with row2_col1:
    comms_input = list(st.multiselect('Select Commodity(s):',st.session_state.comms, key = 'comm_slicer', on_change=item_slicer_update))
# SUB-COMMODITY SLICER (AFFECTED BY RPGs AND COMMODITY)
with row2_col2:
    subcomms_input = list(st.multiselect('Select Sub-Commodity(s):',st.session_state.subcomms, key = 'subcomm_slicer', on_change=item_slicer_update)) 
# RPG SLICER
with row2_col3:
    rpgs_input = list(st.multiselect('Select Retail Price Group(s):',st.session_state.rpgs, key = 'rpg_slicer', on_change=item_slicer_update))   
# ITEM SLICER (AFFECTED BY RPGs AND COMMODITY)
with row2_col4:
    item_input = list(st.multiselect('Select Item(s):',st.session_state['items'], key = 'item_slicer', on_change=item_slicer_update))

row3_col1, row3_col2, row3_col3, row3_col4 = st.columns(4)

# STORE SLICER
with row3_col1:
    store_input = list(st.multiselect('Select Store(s):',st.session_state['stores'],key='store_slicer'))
# CM SLICER
with row3_col2:
    cms_input = list(st.multiselect('Select Category Manager(s):',st.session_state['cms'], key= 'cm_slicer', on_change=item_slicer_update))   
# DEPT SLICER
with row3_col3:
    depts_input = list(st.multiselect('Select Department(s):',st.session_state['depts'], key= 'dept_slicer', on_change=item_slicer_update))  
# VENDOR SLICER
with row3_col4:
    vendor_input = list(st.multiselect('Select Vendor(s):', st.session_state['vendors'], key= 'vendor_slicer', on_change=item_slicer_update))


query_pt1 = '''
SELECT
-- CODE HAS BEEN REMOVED FOR SECURITY REASONS
WHERE 1=1'''

query_pt2 = '''GROUP BY
-- CODE HAS BEEN REMOVED FOR SECURITY REASONS
 '''

# Function to turn list of dates to string of strings
#    ex: "'2024-09-10','2024-08-27'"
def date_list_to_string(dates):
    string = ""
    for a in dates:
        if dates.index(a) == (len(dates) - 1):
            string = string + str(a) +"'"
        elif string == '':
            string = "'"  + str(a) + "','"
        else:
            string = string + str(a) + "','"

    return string

# Create DATE list for Query
if len(date_input) > 0:
    if len(date_input) == 1:
        query_date_input = "-- CODE HAS BEEN REMOVED FOR SECURITY REASONS = '" + str(date_input[0]) + "' "
    elif len(date_input) > 1:
        query_date_input = '-- CODE HAS BEEN REMOVED FOR SECURITY REASONS (' + date_list_to_string(date_input) + ') ' 
else:
    query_date_input = '--'

# Create AD WEEK list for Query
if len(ad_wk_input) > 0:
    if len(ad_wk_input) == 1:
        query_ad_input = "-- CODE HAS BEEN REMOVED FOR SECURITY REASONS = '" + str(ad_wk_input[0]) +"' "
    elif len(ad_wk_input) > 1:
        query_ad_input = '-- CODE HAS BEEN REMOVED FOR SECURITY REASONS (' + date_list_to_string(ad_wk_input) + ') '
else:
    query_ad_input = '--'    

# Create FISCAL WEEK list for Query
if len(fiscal_wk_input) > 0:
    if len(fiscal_wk_input) == 1:
        query_fiscal_input = "-- CODE HAS BEEN REMOVED FOR SECURITY REASONS = '" + str(fiscal_wk_input[0]) +"' "
    elif len(fiscal_wk_input) > 1:
        query_fiscal_input = '-- CODE HAS BEEN REMOVED FOR SECURITY REASONS (' + date_list_to_string(fiscal_wk_input) +') '
else:
    query_fiscal_input = '--'      

# Function to turn list of items into string of strings
#   example: "'011000086','011000087'"
def item_list_to_string(item_list, type):
    if type =='RPG':
        indx = 7
    elif type == 'ITEM':
        indx = 9
    elif type == 'COMM' or type == 'SUBCOMM' or type == 'CM' or type == 'STORE':
        indx = 3
    elif type == 'VENDOR':
        indx = 5

    items_string = ''
    for item in  item_list:
        if type != 'DEPT':
            if item_list.index(item) == (len(item_list) - 1):
                items_string = items_string + item[:indx] + "'"
            elif items_string == '':
                items_string = "'" + item[:indx] + "','"
            else:
                items_string = items_string + item[:indx] + "','"
        else:
            if item_list.index(item) == (len(item_list) - 1):
                items_string = items_string + item + "'"
            elif items_string == '':
                items_string = "'" + item + "','"
            else:
                items_string = items_string + item + "','"
    return items_string


# Create Date list for query

if query_date_input == '--' and query_ad_input != '--':
    gm_query = query_pt1 + query_ad_input 

elif query_date_input == '--' and query_ad_input == '--':
    gm_query = query_pt1 + query_fiscal_input 

elif query_fiscal_input == '--':
    gm_query = query_pt1 + query_date_input




# Create ITEMs list for Query

if len(item_input) > 0:
    if len(item_input) == 1:
        query_item_input = '-- CODE HAS BEEN REMOVED FOR SECURITY REASONS = ' + str(item_input[0][:9]) + ' '
    elif len(item_input) > 1:
        query_item_input = '-- CODE HAS BEEN REMOVED FOR SECURITY REASONS (' + item_list_to_string(item_input,type='ITEM') + ') '
else:
    query_item_input = '--'

# Create COMMODITYs list for Query

if len(comms_input) > 0:
    if len(comms_input) == 1:
        query_comms_input = '-- CODE HAS BEEN REMOVED FOR SECURITY REASONS = ' + str(comms_input[0][:3]) + ' '
    elif len(comms_input) > 1:
        query_comms_input = '-- CODE HAS BEEN REMOVED FOR SECURITY REASONS (' + item_list_to_string(comms_input, type='COMM') +') '
else:
    query_comms_input = '--'

# Create SUB-COMMODITYs list for Query

if len(subcomms_input) > 0:
    if len(subcomms_input) == 1:
        query_subcomms_input = '-- CODE HAS BEEN REMOVED FOR SECURITY REASONS = ' + str(subcomms_input[0][:3]) +' '
    elif len(subcomms_input) > 1:
        query_subcomms_input = '-- CODE HAS BEEN REMOVED FOR SECURITY REASONS (' + item_list_to_string(comms_input, type='SUBCOMM') + ') '
else:
    query_subcomms_input = '--'

# Create RPGs list for Query

if len(rpgs_input) > 0:
    if len(rpgs_input) == 1:
        query_rpg_input = '-- CODE HAS BEEN REMOVED FOR SECURITY REASONS = ' + str(rpgs_input[0][:7]) +' '
    elif len(rpgs_input) > 1:
        query_rpg_input = '-- CODE HAS BEEN REMOVED FOR SECURITY REASONS (' + item_list_to_string(rpgs_input, type='RPG') + ') '
else:
    query_rpg_input = '--'

# Create Store list for Query

if len(store_input) > 0:
    if len(store_input) == 1:
        query_store_input = "-- CODE HAS BEEN REMOVED FOR SECURITY REASONS = '" + str(store_input[0][:3]) + "' "
    elif len(store_input) > 1:
        query_store_input = '-- CODE HAS BEEN REMOVED FOR SECURITY REASONS (' + item_list_to_string(store_input, type='STORE') +') '
else:
    query_store_input = '--'

# Create CMs list for Query

if len(cms_input) > 0:
    if len(cms_input) == 1:
        query_cms_input = "-- CODE HAS BEEN REMOVED FOR SECURITY REASONS = '" + str(cms_input[0][:3]) +"' "
    elif len(cms_input) > 1:
        query_cms_input = '-- CODE HAS BEEN REMOVED FOR SECURITY REASONS (' + item_list_to_string(cms_input, type='CM') + ') '
else:
    query_cms_input = '--'

# Create DEPT list for Query

if len(depts_input) > 0:
    if len(depts_input) == 1:
        query_depts_input = "-- CODE HAS BEEN REMOVED FOR SECURITY REASONS = '" + str(depts_input[0]) +"' "
    elif len(depts_input) > 1:
        query_depts_input = '-- CODE HAS BEEN REMOVED FOR SECURITY REASONS (' + item_list_to_string(depts_input, type='DEPT') + ') '
else:
    query_depts_input = '--'

# Create Vendor list for Query

if len(vendor_input) > 0:
    if len(vendor_input) == 1:
        query_vendor_input = '-- CODE HAS BEEN REMOVED FOR SECURITY REASONS = ' + str(vendor_input[0][:5]) +' '
    elif len(vendor_input) > 1:
        query_vendor_input = '-- CODE HAS BEEN REMOVED FOR SECURITY REASONS (' + item_list_to_string(vendor_input, type='VENDOR') +') '
else:
    query_vendor_input = '--'

# list to include/exclude inputs

query_inputs = [query_item_input , query_comms_input, query_subcomms_input, query_rpg_input, query_depts_input, query_store_input, query_cms_input, query_vendor_input]

for a in query_inputs:
    if a != '--':
        gm_query = gm_query + a

gm_query = gm_query + query_pt2

gm_cols = ['-- CODE HAS BEEN REMOVED FOR SECURITY REASONS']


if 'gm_data' not in st.session_state:
    st.session_state['gm_data'] = pd.DataFrame(data=[])

# Function to apply XGBoost and forecast NET SALES
def forecast_net_sales(group):
    X = group[['DATE_ORDINAL', 'DAY_OF_WEEK', 'MONTH', 'WEEK_OF_MONTH']]
    y = group['NET SALES']
    model = xgb.XGBRegressor(objective='reg:squarederror')
    model.fit(X, y)
    forecast = model.predict(X)
    return pd.Series(forecast, index=group.index)

# Function to forecast the next 7, 14, and 21 days sales for each row using Random Forests
def forecast_next_days_random_forest(group, days):
    X = group[['DATE_ORDINAL', 'DAY_OF_WEEK', 'MONTH', 'WEEK_OF_MONTH']]
    y = group['NET SALES']
    model = RandomForestRegressor(n_estimators=100, max_features='log2', random_state=42)
    model.fit(X, y)
    
    # Forecast for the next 'days' days from each date in the group
    next_days_sales = []
    for date in group['DATE']:
        future_dates = [date + timedelta(days=i) for i in range(1, days + 1)]
        future_dates_ordinal = [date.toordinal() for date in future_dates]
        future_days_of_week = [date.weekday() for date in future_dates]
        future_months = [date.month for date in future_dates]
        future_weeks_of_month = [(date.day - 1) // 7 + 1 for date in future_dates]
        future_X = pd.DataFrame({
            'DATE_ORDINAL': future_dates_ordinal,
            'DAY_OF_WEEK': future_days_of_week,
            'MONTH': future_months,
            'WEEK_OF_MONTH': future_weeks_of_month
        })
        future_forecast = model.predict(future_X)
        next_days_sales.append(np.mean(future_forecast))
    
    return pd.Series(next_days_sales, index=group.index)

st.button('Click to RUN',key='gm_run_button')

try:
    st.write("no error")
except pyodbc.OperationalError:
    st.cache_date.clear()

if st.session_state['gm_run_button']:
    start_time = time.time()
    gm_data = load_data(gm_query,gm_cols)
    end_time = time.time()
    time_elapsed = end_time - start_time
    st.write('RUN TIME:', time.strftime("%H:%M:%S", time.gmtime(time_elapsed)))
    met1, met2, met3, met4 = st.columns(4)
    gm_data['ITEM NO'] = '0' + gm_data['ITEM NO'].astype(str)
    gm_data['RETAIL PRICE GROUP'] = gm_data['RETAIL PRICE GROUP'].fillna(value="0000000")
    st.dataframe(gm_data)
    
    with met1:
        st.metric('UNITS',numerize.numerize(np.sum(gm_data['UNITS'])))
    with met2: 
        st.metric('NET SALES','{:,}'.format(np.round(np.sum(gm_data['NET SALES'].astype(int)))))
    with met3:
        st.metric('GROSS PROFIT','{:,}'.format(np.round(np.sum(gm_data['GROSS PROFIT'].astype(int)))))
    with met4:
        st.metric('MARGIN %', '{:.0%}'.format(np.round((np.sum(gm_data['GROSS PROFIT'].astype(int)) / np.sum(gm_data['NET SALES'].astype(int))),2)))

    df = gm_data.groupby(['-- CODE HAS BEEN REMOVED TO PROTECT FOR SECURITY REASONS']).agg({
           '-- CODE HAS BEEN REMOVED TO PROTECT FOR SECURITY REASONS'
       }).reset_index()

    df['DATE'] = pd.to_datetime(df['AD WEEK'])
    df['DATE_ORDINAL'] = df['DATE'].map(pd.Timestamp.toordinal)
    df['DAY_OF_WEEK'] = df['DATE'].dt.dayofweek
    df['MONTH'] = df['DATE'].dt.month
    df['WEEK_OF_MONTH'] = df['DATE'].apply(lambda x: (x.day - 1) // 7 + 1)
    
    df['FORECASTED NET SALES'] = df.groupby(['ITEM NO', 'AD', 'CY vs PY', 'WIC','ITEM COST']).apply(forecast_net_sales).reset_index(level=[0, 1, 2, 3, 4], drop=True)

    # Apply the next 7, 14, and 21 days forecast function to each group and create new columns with the results
    df['NEXT 7 DAYS SALES'] = df.groupby(['ITEM NO', 'AD', 'CY vs PY','WIC','ITEM COST']).apply(lambda group: forecast_next_days_random_forest(group, 7)).reset_index(level=[0, 1, 2, 3, 4], drop=True)
    # df['NEXT 14 DAYS SALES'] = df.groupby(['ITEM NO', 'AD', 'CY vs PY']).apply(lambda group: forecast_next_days_random_forest(group, 14)).reset_index(level=[0, 1, 2], drop=True)
    # df['NEXT 21 DAYS SALES'] = df.groupby(['ITEM NO', 'AD', 'CY vs PY']).apply(lambda group: forecast_next_days_random_forest(group, 21)).reset_index(level=[0, 1, 2], drop=True)
    
    df = df.drop(columns = ['-- CODE HAS BEEN REMOVED TO PROTECT FOR SECURITY REASONS'])
    st.write('Forecasted Next Ad WEEK Net Sales')
    st.dataframe(df)

    if st.session_state['ad_slicer'] != []:
        if len(st.session_state['ad_slicer']) == 1 or len(st.session_state['ad_slicer']) == 2:
            temp_df = gm_data[gm_data['AD WEEK'].isin(st.session_state.ad_slicer)].groupby(['AD WEEK','DATE']).agg({'UNITS':'sum','NET SALES':'sum'}).reset_index()
            temp_df['AD WEEK'] = pd.to_datetime(temp_df['AD WEEK']).dt.date
            temp_df['DATE'] = pd.to_datetime(temp_df['DATE']).dt.date
            temp_df['UNITS'] = pd.to_numeric(temp_df['UNITS'])
            temp_df['NET SALES'] = pd.to_numeric(temp_df['NET SALES'])
            x = 'DATE'
            # y = temp_df['UNITS']
            # y2 = temp_df['NET SALES']
            # x = pd.to_datetime(gm_data['DATE'][gm_data['AD WEEK'].isin(st.session_state.ad_slicer)].unique()).date
            # y = gm_data['UNITS'][gm_data['AD WEEK'].isin(st.session_state.ad_slicer)].groupby(gm_data['DATE']).sum()
            # y2 = gm_data['NET SALES'][gm_data['AD WEEK'].isin(st.session_state.ad_slicer)].groupby(gm_data['DATE']).sum()
        elif len(st.session_state['ad_slicer']) > 2:
            temp_df = gm_data[gm_data['AD WEEK'].isin(st.session_state.ad_slicer)].groupby(['AD WEEK']).agg({'UNITS':'sum','NET SALES':'sum'}).reset_index()
            temp_df['AD WEEK'] = pd.to_datetime(temp_df['AD WEEK']).dt.date
            #temp_df['DATE'] = pd.to_datetime(temp_df['DATE']).dt.date
            temp_df['UNITS'] = pd.to_numeric(temp_df['UNITS'])
            temp_df['NET SALES'] = pd.to_numeric(temp_df['NET SALES'])
            x = 'AD WEEK'
            # y = temp_df['UNITS']
            # y2 = temp_df['NET SALES']
            # x = pd.to_datetime(gm_data['AD WEEK'][gm_data['AD WEEK'].isin(st.session_state.ad_slicer)].unique()).date
            # y = gm_data['UNITS'][gm_data['AD WEEK'].isin(st.session_state.ad_slicer)].groupby(gm_data['AD WEEK']).sum()
            # y2 = gm_data['NET SALES'][gm_data['AD WEEK'].isin(st.session_state.ad_slicer)].groupby(gm_data['AD WEEK']).sum()
    elif st.session_state['fiscal_slicer'] != []:
        if len(st.session_state['fiscal_slicer']) == 1 or len(st.session_state['fiscal_slicer']) == 2:
            temp_df = gm_data[gm_data['FISCAL WEEK'].isin(st.session_state.fiscal_slicer)].groupby(['FISCAL WEEK','DATE']).agg({'UNITS':'sum','NET SALES':'sum'}).reset_index()
            temp_df['FISCAL WEEK'] = pd.to_datetime(temp_df['FISCAL WEEK']).dt.date
            temp_df['DATE'] = pd.to_datetime(temp_df['DATE']).dt.date
            temp_df['UNITS'] = pd.to_numeric(temp_df['UNITS'])
            temp_df['NET SALES'] = pd.to_numeric(temp_df['NET SALES'])
            x = 'DATE'
            # y = temp_df['UNITS']
            # y2 = temp_df['NET SALES']
            # x = pd.to_datetime(gm_data['DATE'][gm_data['FISCAL WEEK'].isin(st.session_state.fiscal_slicer)].unique()).date
            # #x = gm_data['DATE'][gm_data['FISCAL WEEK'].isin(st.session_state.fiscal_slicer)]
            # y = gm_data['UNITS'][gm_data['FISCAL WEEK'].isin(st.session_state.fiscal_slicer)].groupby(gm_data['DATE']).sum()
            # y2 = gm_data['NET SALES'][gm_data['FISCAL WEEK'].isin(st.session_state.fiscal_slicer)].groupby(gm_data['DATE']).sum()
        elif len(st.session_state['fiscal_slicer']) > 2:
            temp_df = gm_data[gm_data['FISCAL WEEK'].isin(st.session_state.fiscal_slicer)].groupby(['FISCAL WEEK']).agg({'UNITS':'sum','NET SALES':'sum'}).reset_index()
            temp_df['FISCAL WEEK'] = pd.to_datetime(temp_df['FISCAL WEEK']).dt.date
            #temp_df['DATE'] = pd.to_datetime(temp_df['DATE']).dt.date
            temp_df['UNITS'] = pd.to_numeric(temp_df['UNITS'])
            temp_df['NET SALES'] = pd.to_numeric(temp_df['NET SALES'])
            x = 'FISCAL WEEK'
            # y = temp_df['UNITS']
            # y2 = temp_df['NET SALES']
            # x = pd.to_datetime(gm_data['FISCAL WEEK'][gm_data['FISCAL WEEK'].isin(st.session_state.fiscal_slicer)].unique()).date
            # y = gm_data['UNITS'][gm_data['FISCAL WEEK'].isin(st.session_state.fiscal_slicer)].groupby(gm_data['FISCAL WEEK']).sum()
            # y2 = gm_data['NET SALES'][gm_data['FISCAL WEEK'].isin(st.session_state.fiscal_slicer)].groupby(gm_data['FISCAL WEEK']).sum()

    else:
        temp_df = gm_data
        temp_df['DATE'] = pd.to_datetime(temp_df['DATE']).dt.date
        #temp_df['DATE'] = pd.to_datetime(temp_df['DATE']).dt.date
        temp_df['UNITS'] = pd.to_numeric(temp_df['UNITS'])
        temp_df['NET SALES'] = pd.to_numeric(temp_df['NET SALES'])
        x = 'DATE'


    #y = x['UNITS']
    # fig1, ax1 = plt.figure(facecolor='white' , figsize=(10,6))
    # #spec = fig1.add_gridspec(ncols=1, nrows= 2)
    # #ax0 = fig1.add_subplot(spec[0,0])
    # #ax1 = fig.add_subplot(spec[1,0])
    # ax1.plot(x,y)
    # #ax0.set_label(axis ='y', label = '')
    # ax1.tick_params(axis= 'x', labelsize = 8,labelrotation = 35)
    # #ax0.secondary_yaxis(y2)
    # #ax1.plot(x,y2)
    # #ax1.tick_params(axis= 'x', labelsize = 8,labelrotation = 35)

    # fig2, ax2 = plt.figure(facecolor='white', figsize=(10,6))
    # ax2.plot(x,y2)
    # ax2.tick_params(axis= 'x', labelsize = 8,labelrotation = 35)
    # plt.xlabel('Dates/Weeks')
    # plt.ylabel('Units/Net Sales')

# ****create aggregated dataframe for trend lines??
    # st.dataframe(temp_df)
    # st.write(temp_df['UNITS'].dtype)
    # st.write(temp_df['NET SALES'].dtype)

# line chart with streamlit plotting
    try:
        if len(temp_df['DATE'].unique()) >= 7:
            #st.line_chart(temp_df,x=x,y=['UNITS','NET SALES'])

            chart1 = alt.Chart(temp_df).mark_line(color='#0077bb', point = True).encode(
            x=x,
            y=alt.Y('UNITS', axis=alt.Axis(title='Units', titleColor='#0077bb'))
            ).properties(
            height=400
            )

            chart2 = alt.Chart(temp_df).mark_line(color='#aec7e8', point = True).encode(
            x=x,
            y=alt.Y('NET SALES', axis=alt.Axis(title='Net Sales', titleColor='#aec7e8'))
            ).properties(
            height=400
            , title = 'Units/Sales Trend'
            )

            # Combine the two charts and resolve the scale
            combined_chart = alt.layer(chart1, chart2).resolve_scale(
            y='independent'
            )

            # Display the chart in Streamlit
            st.altair_chart(combined_chart, use_container_width=True)
    except KeyError:
        if len(temp_df['AD WEEK'].unique()) >= 2 or len(temp_df['FISCAL WEEK'].unique()) >= 2:
            #st.line_chart(temp_df,x=x,y=['UNITS','NET SALES'])
            
            chart1 = alt.Chart(temp_df).mark_line(color='#0077bb', point = True).encode(
            x=x,
            y=alt.Y('UNITS', axis=alt.Axis(title='Units', titleColor='#0077bb'))
            ).properties(
            height=400
            )

            chart2 = alt.Chart(temp_df).mark_line(color='#aec7e8', point = True).encode(
            x=x,
            y=alt.Y('NET SALES', axis=alt.Axis(title='Net Sales', titleColor='#aec7e8'))
            ).properties(
            height=400
            )

            # Combine the two charts and resolve the scale
            combined_chart = alt.layer(chart1, chart2).resolve_scale(
            y='independent'
            )

            # Display the chart in Streamlit
            st.altair_chart(combined_chart, use_container_width=True)

# line chart with matplotlib.pyplot plotting
    # try:
    #     if len(temp_df['DATE'].unique()) >= 7: # need to show at least a week of data 
    #         fig, axs = plt.subplots(2)
    #         temp_df.plot(x= x, y='UNITS', kind= 'line',ax = axs[0])
    #         #axs[0].plot(temp_df, x,y)
    #         axs[0].set_title('UNIT TREND')
    #         axs[0].set_ylabel('UNITS')
    #         axs[0].tick_params(axis='x',labelsize = 8, labelrotation = 35)
    #         temp_df.plot(x=x, y='NET SALES', kind= 'line', ax=axs[1])
    #         #axs[1].plot(temp_df, x,y2, 'tab:orange')
    #         axs[1].set_title('SALES TREND')
    #         axs[1].set_ylabel('NET SALES')
    #         axs[1].tick_params(axis='x',labelsize = 8, labelrotation = 35)
    #         #axs[1].ticklabel_format(style = 'plain')

    #         formatter = ticker.ScalarFormatter()
    #         formatter.set_scientific(False)
    #         axs[1].yaxis.set_major_formatter(formatter)

    #         for ax in fig.get_axes():
    #             ax.label_outer()

    #         st.pyplot(fig)
    # except KeyError:
        if len(temp_df['AD WEEK'].unique()) >= 2 or len(temp_df['FISCAL WEEK'].unique()) >= 2: # need to show at least a week of data 
            fig, axs = plt.subplots(2)
            temp_df.plot(x= x, y='UNITS', kind= 'line',ax = axs[0])
            #axs[0].plot(temp_df, x,y)
            axs[0].set_title('UNIT TREND')
            axs[0].set_ylabel('UNITS')
            axs[0].tick_params(axis='x',labelsize = 8, labelrotation = 35)
            temp_df.plot(x=x, y='NET SALES', kind= 'line', ax=axs[1])
            #axs[1].plot(temp_df, x,y2, 'tab:orange')
            axs[1].set_title('SALES TREND')
            axs[1].set_ylabel('NET SALES')
            axs[1].tick_params(axis='x',labelsize = 8, labelrotation = 35)
            #axs[1].ticklabel_format(style = 'plain')

            formatter = ticker.ScalarFormatter()
            formatter.set_scientific(False)
            axs[1].yaxis.set_major_formatter(formatter)

            for ax in fig.get_axes():
                ax.label_outer()

            st.pyplot(fig)

# bar chart for dept with streamlit plotting
    if len(gm_data['DEPT'].unique()) > 1:
        df_bar_dept = pd.DataFrame({
            'DEPT': list(gm_data['DEPT'].unique())
            , 'UNITS' : list(gm_data['UNITS'].groupby(gm_data['DEPT']).sum())
            , 'NET SALES' : list(gm_data['NET SALES'].groupby(gm_data['DEPT']).sum())
        })
        df_bar_dept['UNITS'] = pd.to_numeric(df_bar_dept['UNITS'])
        df_bar_dept['NET SALES'] = pd.to_numeric(df_bar_dept['NET SALES'])
        #st.bar_chart(df_bar_dept, x='DEPT', y=['UNITS','NET SALES'])

        chart = alt.Chart(df_bar_dept).transform_fold(
            ['UNITS', 'NET SALES'],
            as_=['Measure', 'Value']
        ).mark_bar().encode(
            x='DEPT:N',
            y=alt.Y('Value:Q',stack='zero'),
            color='Measure:N'
        ).properties(
            height=400,
            title='Units and Net Sales by Department'
        )

        # Display the chart in Streamlit
        st.altair_chart(chart, use_container_width=True)

# bar chart for dept with matplotlib.pyplot plotting
    # if len(gm_data['DEPT'].unique()) > 1:

        # fig_bar_dept = plt.figure(figsize=(10,5))
        # ax = plt.subplot()
        # ax.tick_params(axis='x',labelsize = 8, labelrotation = 35)

        # formatter = ticker.ScalarFormatter()
        # formatter.set_scientific(False)
        # ax.yaxis.set_major_formatter(formatter)

        # df_bar_dept = pd.DataFrame({
        #     'DEPT': list(gm_data['DEPT'].unique())
        #     , 'UNITS' : list(gm_data['UNITS'].groupby(gm_data['DEPT']).sum())
        #     , 'NET SALES' : list(gm_data['NET SALES'].groupby(gm_data['DEPT']).sum())
        # })
        # df_bar_dept['UNITS'] = pd.to_numeric(df_bar_dept['UNITS'])
        # df_bar_dept['NET SALES'] = pd.to_numeric(df_bar_dept['NET SALES'])
            
        # tot_labels_units = []
        # tot_labels_sales = []
        # for a in list(df_bar_dept['DEPT'].unique()):
            
        #     b = float(df_bar_dept['UNITS'][df_bar_dept['DEPT']==a].sum())
        #     tot = numerize.numerize(b)
        #     tot_labels_units.append(tot)
        #     c = float(df_bar_dept['NET SALES'][df_bar_dept['DEPT']==a].sum())
        #     tot = numerize.numerize(c)
        #     tot_labels_sales.append(tot)


        # a = ax.bar(df_bar_dept['DEPT'], df_bar_dept['NET SALES'])
        # ax.bar_label(a, label_type='edge',labels = tot_labels_sales)
        
        # a = ax.bar(df_bar_dept['DEPT'], df_bar_dept['UNITS'], color = 'orange')
        # ax.bar_label(a, label_type='edge', labels =  tot_labels_units)

        # plt.legend(['NET SALES','UNITS'],loc = 'upper right')
        # plt.xlabel('DEPARTMENT')
        # plt.ylabel('Amount in UNITS/NET SALES')
        # st.pyplot(fig_bar_dept)

# bar chart for CM with streamlit plotting
    if len(gm_data['CM'].unique()) > 1 and len(gm_data['CM'].unique()) <= 10:
        df_bar_cm = pd.DataFrame({
            'CM': list(gm_data['CM'].unique())
            , 'UNITS' : list(gm_data['UNITS'].groupby(gm_data['CM']).sum())
            , 'NET SALES' : list(gm_data['NET SALES'].groupby(gm_data['CM']).sum())
        })
        df_bar_cm['UNITS'] = pd.to_numeric(df_bar_cm['UNITS'])
        df_bar_cm['NET SALES'] = pd.to_numeric(df_bar_cm['NET SALES'])
        #st.bar_chart(df_bar_cm, x='CM', y=['UNITS','NET SALES'])

        chart = alt.Chart(df_bar_cm).transform_fold(
            ['UNITS', 'NET SALES'],
            as_=['Measure', 'Value']
        ).mark_bar().encode(
            x='CM:N',
            y=alt.Y('Value:Q',stack='zero'),
            color='Measure:N'
        ).properties(
            height=400,
            title='Units and Net Sales by CM'
        )

        # Display the chart in Streamlit
        st.altair_chart(chart, use_container_width=True)

# bar chart for CM with matplotlib.pyplot plotting
    # if len(gm_data['CM'].unique()) > 1 and len(gm_data['CM'].unique()) <= 10:

        # fig_bar_cm = plt.figure(figsize=(10,5))
        # ax = plt.subplot()
        # ax.tick_params(axis='x',labelsize = 8, labelrotation = 35)

        # formatter = ticker.ScalarFormatter()
        # formatter.set_scientific(False)
        # ax.yaxis.set_major_formatter(formatter)

        # df_bar_cm = pd.DataFrame({
        #     'CM': list(gm_data['CM'].unique())
        #     , 'UNITS' : list(gm_data['UNITS'].groupby(gm_data['CM']).sum())
        #     , 'NET SALES' : list(gm_data['NET SALES'].groupby(gm_data['CM']).sum())
        # })
        # df_bar_cm['UNITS'] = pd.to_numeric(df_bar_cm['UNITS'])
        # df_bar_cm['NET SALES'] = pd.to_numeric(df_bar_cm['NET SALES'])
            
        # tot_labels_units = []
        # tot_labels_sales = []
        # for a in list(df_bar_cm['CM'].unique()):
            
        #     b = float(df_bar_cm['UNITS'][df_bar_cm['CM']==a].sum())
        #     tot = numerize.numerize(b)
        #     tot_labels_units.append(tot)
        #     c = float(df_bar_cm['NET SALES'][df_bar_cm['CM']==a].sum())
        #     tot = numerize.numerize(c)
        #     tot_labels_sales.append(tot)


        # a = ax.bar(df_bar_cm['CM'], df_bar_cm['NET SALES'])
        # ax.bar_label(a, label_type='edge',labels = tot_labels_sales)
        
        # a = ax.bar(df_bar_cm['CM'], df_bar_cm['UNITS'], color = 'orange')
        # ax.bar_label(a, label_type='edge', labels =  tot_labels_units)

        # plt.legend(['NET SALES','UNITS'],loc = 'upper right')
        # plt.xlabel('CM')
        # plt.ylabel('Amount in UNITS/NET SALES')
        # st.pyplot(fig_bar_cm)

# bar chart for Commodity with streamlit plotting
    if len(gm_data['COMMODITY'].unique()) <= 10:
        df_bar_comm = pd.DataFrame({
            'COMMODITY': list(gm_data['COMMODITY'].unique())
            , 'UNITS' : list(gm_data['UNITS'].groupby(gm_data['COMMODITY']).sum())
            , 'NET SALES' : list(gm_data['NET SALES'].groupby(gm_data['COMMODITY']).sum())
        })
        df_bar_comm['UNITS'] = pd.to_numeric(df_bar_comm['UNITS'])
        df_bar_comm['NET SALES'] = pd.to_numeric(df_bar_comm['NET SALES'])
        #st.bar_chart(df_bar_comm, x='COMMODITY', y=['UNITS','NET SALES'])

        chart = alt.Chart(df_bar_comm).transform_fold(
            ['UNITS', 'NET SALES'],
            as_=['Measure', 'Value']
        ).mark_bar().encode(
            x='COMMODITY:N',
            y=alt.Y('Value:Q',stack='zero'),
            color='Measure:N'
        ).properties(
            height=400,
            title='Units and Net Sales by Commodity'
        )

        # Display the chart in Streamlit
        st.altair_chart(chart, use_container_width=True)

# bar chart for Commodity with matplotlib.pyplot plotting
    # if len(gm_data['COMMODITY'].unique()) <= 10:

        # fig_bar_comm = plt.figure(figsize=(10,5))
        # ax = plt.subplot()
        # ax.tick_params(axis='x',labelsize = 8, labelrotation = 35)

        # formatter = ticker.ScalarFormatter()
        # formatter.set_scientific(False)
        # ax.yaxis.set_major_formatter(formatter)

        # df_bar_comm = pd.DataFrame({
        #     'COMMODITY': list(gm_data['COMMODITY'].unique())
        #     , 'UNITS' : list(gm_data['UNITS'].groupby(gm_data['COMMODITY']).sum())
        #     , 'NET SALES' : list(gm_data['NET SALES'].groupby(gm_data['COMMODITY']).sum())
        # })
        # df_bar_comm['UNITS'] = pd.to_numeric(df_bar_comm['UNITS'])
        # df_bar_comm['NET SALES'] = pd.to_numeric(df_bar_comm['NET SALES'])
            
        # tot_labels_units = []
        # tot_labels_sales = []
        # for a in list(df_bar_comm['COMMODITY'].unique()):
            
        #     b = float(df_bar_comm['UNITS'][df_bar_comm['COMMODITY']==a].sum())
        #     tot = numerize.numerize(b)
        #     tot_labels_units.append(tot)
        #     c = float(df_bar_comm['NET SALES'][df_bar_comm['COMMODITY']==a].sum())
        #     tot = numerize.numerize(c)
        #     tot_labels_sales.append(tot)


        # a = ax.bar(df_bar_comm['COMMODITY'], df_bar_comm['NET SALES'])
        # ax.bar_label(a, label_type='edge',labels = tot_labels_sales)
        
        # a = ax.bar(df_bar_comm['COMMODITY'], df_bar_comm['UNITS'], color = 'orange')
        # ax.bar_label(a, label_type='edge', labels =  tot_labels_units)

        # plt.legend(['NET SALES','UNITS'],loc = 'upper right')
        # plt.xlabel('COMMODITY')
        # plt.ylabel('Amount in UNITS/NET SALES')
        # st.pyplot(fig_bar_comm)

# bar chart for Sub-Commodity with streamlit plotting
    if len(gm_data['SUB-COMMODITY'].unique()) > 1 and len(gm_data['SUB-COMMODITY'].unique()) <= 10:
        df_bar_subcomm = pd.DataFrame({
            'SUB-COMMODITY': list(gm_data['SUB-COMMODITY'].unique())
            , 'UNITS' : list(gm_data['UNITS'].groupby(gm_data['SUB-COMMODITY']).sum())
            , 'NET SALES' : list(gm_data['NET SALES'].groupby(gm_data['SUB-COMMODITY']).sum())
        })
        df_bar_subcomm['UNITS'] = pd.to_numeric(df_bar_subcomm['UNITS'])
        df_bar_subcomm['NET SALES'] = pd.to_numeric(df_bar_subcomm['NET SALES'])
        #st.bar_chart(df_bar_subcomm, x='SUB-COMMODITY', y=['UNITS','NET SALES'])

        chart = alt.Chart(df_bar_subcomm).transform_fold(
            ['UNITS', 'NET SALES'],
            as_=['Measure', 'Value']
        ).mark_bar().encode(
            x='SUB-COMMODITY:N',
            y=alt.Y('Value:Q',stack='zero'),
            color='Measure:N'
        ).properties(
            height=400,
            title='Units and Net Sales by Sub-Commodity'
        )

        # Display the chart in Streamlit
        st.altair_chart(chart, use_container_width=True)

# bar chart for Sub-Commodity with matplotlib.pyplot plotting
    # if len(gm_data['SUB-COMMODITY'].unique()) > 1 and len(gm_data['SUB-COMMODITY'].unique()) <= 10:

        # fig_bar_subcomm = plt.figure(figsize=(10,5))
        # ax = plt.subplot()
        # ax.tick_params(axis='x',labelsize = 8, labelrotation = 35)

        # formatter = ticker.ScalarFormatter()
        # formatter.set_scientific(False)
        # ax.yaxis.set_major_formatter(formatter)

        # df_bar_subcomm = pd.DataFrame({
        #     'SUB-COMMODITY': list(gm_data['SUB-COMMODITY'].unique())
        #     , 'UNITS' : list(gm_data['UNITS'].groupby(gm_data['SUB-COMMODITY']).sum())
        #     , 'NET SALES' : list(gm_data['NET SALES'].groupby(gm_data['SUB-COMMODITY']).sum())
        # })
        # df_bar_subcomm['UNITS'] = pd.to_numeric(df_bar_subcomm['UNITS'])
        # df_bar_subcomm['NET SALES'] = pd.to_numeric(df_bar_subcomm['NET SALES'])
            
        # tot_labels_units = []
        # tot_labels_sales = []
        # for a in list(df_bar_subcomm['SUB-COMMODITY'].unique()):
            
        #     b = float(df_bar_subcomm['UNITS'][df_bar_subcomm['SUB-COMMODITY']==a].sum())
        #     tot = numerize.numerize(b)
        #     tot_labels_units.append(tot)
        #     c = float(df_bar_subcomm['NET SALES'][df_bar_subcomm['SUB-COMMODITY']==a].sum())
        #     tot = numerize.numerize(c)
        #     tot_labels_sales.append(tot)


        # a = ax.bar(df_bar_subcomm['SUB-COMMODITY'], df_bar_subcomm['NET SALES'])
        # ax.bar_label(a, label_type='edge',labels = tot_labels_sales)
        
        # a = ax.bar(df_bar_subcomm['SUB-COMMODITY'], df_bar_subcomm['UNITS'], color = 'orange')
        # ax.bar_label(a, label_type='edge', labels =  tot_labels_units)

        # plt.legend(['NET SALES','UNITS'],loc = 'upper right')
        # plt.xlabel('SUB-COMMODITY')
        # plt.ylabel('Amount in UNITS/NET SALES')
        # st.pyplot(fig_bar_subcomm)

# bar chart for RPG with streamlit plotting
    if len(gm_data['RETAIL PRICE GROUP'].unique()) > 1 and len(gm_data['RETAIL PRICE GROUP'].unique()) <= 10:
        df_bar_rpg = pd.DataFrame({
            'RETAIL PRICE GROUP': list(gm_data['RETAIL PRICE GROUP'].unique())
            , 'UNITS' : list(gm_data['UNITS'].groupby(gm_data['RETAIL PRICE GROUP']).sum())
            , 'NET SALES' : list(gm_data['NET SALES'].groupby(gm_data['RETAIL PRICE GROUP']).sum())
        })
        df_bar_rpg['UNITS'] = pd.to_numeric(df_bar_rpg['UNITS'])
        df_bar_rpg['NET SALES'] = pd.to_numeric(df_bar_rpg['NET SALES'])
        #st.bar_chart(df_bar_rpg, x='RETAIL PRICE GROUP', y=['UNITS','NET SALES'])

        chart = alt.Chart(df_bar_rpg).transform_fold(
            ['UNITS', 'NET SALES'],
            as_=['Measure', 'Value']
        ).mark_bar().encode(
            x='RETAIL PRICE GROUP:N',
            y=alt.Y('Value:Q',stack='zero'),
            color='Measure:N'
        ).properties(
            height=400,
            title='Units and Net Sales by Retail Price Group'
        )

        # Display the chart in Streamlit
        st.altair_chart(chart, use_container_width=True)

# bar chart for RPG with matplotlib.pyplot plotting
    # if len(gm_data['RETAIL PRICE GROUP'].unique()) > 1 and len(gm_data['RETAIL PRICE GROUP'].unique()) <= 10:

        # fig_bar_rpg = plt.figure(figsize=(10,5))
        # ax = plt.subplot()
        # ax.tick_params(axis='x',labelsize = 8, labelrotation = 35)

        # formatter = ticker.ScalarFormatter()
        # formatter.set_scientific(False)
        # ax.yaxis.set_major_formatter(formatter)

        # df_bar_rpg = pd.DataFrame({
        #     'RETAIL PRICE GROUP': list(gm_data['RETAIL PRICE GROUP'].unique())
        #     , 'UNITS' : list(gm_data['UNITS'].groupby(gm_data['RETAIL PRICE GROUP']).sum())
        #     , 'NET SALES' : list(gm_data['NET SALES'].groupby(gm_data['RETAIL PRICE GROUP']).sum())
        # })
        # df_bar_rpg['UNITS'] = pd.to_numeric(df_bar_rpg['UNITS'])
        # df_bar_rpg['NET SALES'] = pd.to_numeric(df_bar_rpg['NET SALES'])
            
        # tot_labels_units = []
        # tot_labels_sales = []
        # for a in list(df_bar_rpg['RETAIL PRICE GROUP'].unique()):
            
        #     b = float(df_bar_rpg['UNITS'][df_bar_rpg['RETAIL PRICE GROUP']==a].sum())
        #     tot = numerize.numerize(b)
        #     tot_labels_units.append(tot)
        #     c = float(df_bar_rpg['NET SALES'][df_bar_rpg['RETAIL PRICE GROUP']==a].sum())
        #     tot = numerize.numerize(c)
        #     tot_labels_sales.append(tot)


        # a = ax.bar(df_bar_rpg['RETAIL PRICE GROUP'], df_bar_rpg['NET SALES'])
        # ax.bar_label(a, label_type='edge',labels = tot_labels_sales)
        
        # a = ax.bar(df_bar_rpg['RETAIL PRICE GROUP'], df_bar_rpg['UNITS'], color = 'orange')
        # ax.bar_label(a, label_type='edge', labels =  tot_labels_units)

        # plt.legend(['NET SALES','UNITS'],loc = 'upper right')
        # plt.xlabel('RETAIL PRICE GROUP')
        # plt.ylabel('Amount in UNITS/NET SALES')
        # st.pyplot(fig_bar_rpg)

#st.session_state