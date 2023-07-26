import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf 
import seaborn as sns
import matplotlib.dates as mdates
from tensorflow.keras.models import load_model
import math 
from datetime import date, timedelta, datetime 
from pandas.plotting import register_matplotlib_converters 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates 
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Dropout 
from tensorflow.keras.callbacks import EarlyStopping 
from sklearn.preprocessing import RobustScaler, MinMaxScaler 
import tensorflow as tf
import seaborn as sns # Visualization
sns.set_style('white', { 'axes.spines.right': False, 'axes.spines.top': False})
import io
import datetime

tickers = ('HDFCBANK.NS','TCS.NS','POWERGRID.NS','BAJAJ-AUTO.NS')
time_intervals=["DAILY","15 MINUTES"]
time_intervalss=["Daily","15 minutes"]

period = "60d"
interval = "15m"
##claulte return for sccaling  the close fdor different stocks
def retrn(df):
    r1=df.pct_change()
    cumr=(1+r1).cumprod()-1
    cumr= cumr.fillna(0)
    return cumr

def plot_model(model,df,stocks,Price) :
    FEATURES = ['Open','High','Low','Close','Adj Close','Volume']
    data = pd.DataFrame(df)
    data_filtered = data[FEATURES]
# We add a prediction column and set dummy values to prepare the data for scaling
    data_filtered_ext = data_filtered.copy()
    data_filtered_ext['Prediction'] = data_filtered_ext[Price]


# Get the number of rows in the data
    nrows = data_filtered.shape[0]

# Convert the data to numpy values
    np_data_unscaled = np.array(data_filtered)
    np_data = np.reshape(np_data_unscaled, (nrows, -1))
# Transform the data by scaling each feature to a range between 0 and 1
    scaler = MinMaxScaler()
    np_data_scaled = scaler.fit_transform(np_data_unscaled)

# Creating a separate scaler that works on a single column for scaling predictions
    scaler_pred = MinMaxScaler()
    df_Close = pd.DataFrame(data_filtered_ext[Price])
    np_Close_scaled = scaler_pred.fit_transform(df_Close)

# Set the sequence length - this is the timeframe used to make a single prediction
    sequence_length = 50

# Prediction Index
    index_Close = data.columns.get_loc(Price)

# Split the training data into train and train data sets
# As a first step, we get the number of rows to train the model on 80% of the data
    train_data_len = math.ceil(np_data_scaled.shape[0] * 0.8)

# Create the training and test data
    train_data = np_data_scaled[0:train_data_len, :]
    test_data = np_data_scaled[train_data_len - sequence_length:, :]

# The RNN needs data with the format of [samples, time steps, features]
# Here, we create N samples, sequence_length time steps per sample, and 6 features
    def partition_dataset(sequence_length, data):
        x, y = [], []
        data_len = data.shape[0]
        for i in range(sequence_length, data_len):
            x.append(data[i-sequence_length:i,:]) #contains sequence_length values 0-sequence_length * columsn
            y.append(data[i, index_Close]) #contains the prediction values for validation,  for single-step prediction

    # Convert the x and y to numpy arrays
        x = np.array(x)
        y = np.array(y)
        return x, y


# Generate training data and test data
    x_train, y_train = partition_dataset(sequence_length, train_data)
    x_test, y_test = partition_dataset(sequence_length, test_data)


    y_pred = model.predict(x_test)
    y_pred_reshaped = y_pred.reshape(-1, 1)
    y_pred_rescaled = scaler_pred.inverse_transform(y_pred_reshaped)
    y_pred_rescaled = y_pred_rescaled[:len(y_test)]
    train = pd.DataFrame(data_filtered_ext[Price][:train_data_len + 1]).rename(columns={Price: 'y_train'})
    valid = pd.DataFrame(data_filtered_ext[Price][train_data_len:]).rename(columns={Price: 'y_test'})
    valid["y_pred"]= y_pred_rescaled
    valid.insert(1, "residuals", valid["y_pred"] - valid["y_test"], True)
#Unscale the predicted values
    y_pred = y_pred_rescaled
    y_test_unscaled = scaler_pred.inverse_transform(y_test.reshape(-1, 1))
    
    # Mean Absolute Error (MAE)
    MAE = np.round(mean_absolute_error(y_test_unscaled, y_pred),2)

# Mean Absolute Percentage Error (MAPE)
    MAPE = np.round(np.mean((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled))) * 100,2)
    

# Median Absolute Percentage Error (MDAPE)
    MDAPE = np.round(np.median((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled)) ) * 100,2)
    


# Zoom in to a closer timeframe
    display_start_date = "2019-01-01"


    df_union = pd.concat([train, valid])

# Zoom in to a closer timeframe
    df_union_zoom = df_union[df_union.index > display_start_date]

    fig, ax1 = plt.subplots(figsize=(16, 8))
    plt.title(f"Train vs Test and Prediction Plot for {stocks}")
    plt.ylabel(f"{price} Price", fontsize=18)
    sns.set_palette(["#090364", "#1960EF", "#EF5919"])
    sns.lineplot(data=df_union_zoom[['y_pred', 'y_train', 'y_test']], linewidth=1.0, dashes=False, ax=ax1)
    plt.legend()
    
    df_temp = df[-250:]
    new_df = df_temp.filter(FEATURES)


# Get the last N day closing price values and scale the data to be values between 0 and 1
    last_N_days = new_df[-250:].values
    last_N_days_scaled = scaler.transform(last_N_days)

# Create an empty list and Append past N days
    X_test_new = []
    X_test_new.append(last_N_days_scaled)

# Convert the X_test data set to a numpy array and reshape the data
    pred_price_scaled = model.predict(np.array(X_test_new).reshape(5,50,6))
    pred_price_unscaled = scaler_pred.inverse_transform(pred_price_scaled.reshape(-1, 1))
    
    price_today = np.round(new_df[Price][-1], 2)
    predicted_price = np.round(pred_price_unscaled.ravel()[0], 2)
    change_percent = np.round(100 - (price_today * 100)/predicted_price, 2)
    
    # Create an empty dataframe
    final= pd.DataFrame()

# Define the number of rows to append
    num_rows = 5

# Perform the loop for each row
    rows = []
    for i in range(num_rows):
    # Define the values for each column in the row
        row_data = {
            "Predictions": np.round(pred_price_unscaled.ravel()[i], 4),  # Example formula for the 'pred' column
            "Percent Change(%)": np.round(100 - (price_today * 100) / np.round(pred_price_unscaled.ravel()[i], 4), 4)  # Example formula for the 'percent' column
        }
        rows.append(row_data)
# Concatenate the rows to the dataframe
    final = pd.concat([final, pd.DataFrame(rows)], ignore_index=True)

    return fig, valid,MAE,MAPE,MDAPE,final


##defining time indices

def get_date(present_time, days=7):
    next_date = []
    for _ in range(days):
        present_time += datetime.timedelta(days=1)
        if present_time.weekday() < 5:  # Exclude Saturday (5) and Sunday (6)
            if len(next_date) == 0 or len(next_date[-1]) == 5:
                next_date.append([present_time.strftime('%Y-%m-%d')])
            else:
                next_date[-1].append(present_time.strftime('%Y-%m-%d'))
    return next_date


def get_time(last_time_point, minutes=15):
    next_time = []
    for _ in range(5):
        last_time_point += datetime.timedelta(minutes=minutes)
        next_time.append(last_time_point.strftime("%Y-%m-%d %H:%M:%S"))
    return next_time


# Sidebar sections
st.sidebar.title('Sections')
section = st.sidebar.radio('Go to', ['Project Summary', 'Analysis', 'Prediction'])

# Section: Intro
if section == 'Project Summary':
    
    st.title('Multivariate Stock prediction')
    st.header('Obejectives of my study :')
    st.subheader('1. Provide investors with insights into the future performance of stocks based on real-time market data.')
    st.subheader('2. Carrying out  univariate and multivariate analysis for forecasting and comparing their results.')
    st.subheader("3.  Identifying which modeling techniques would be a better choice for achieving the best possible stock forecasting.")
    st.subheader('4. Comparing the forecasting accuracy using   2 different target variable i.e., Close and Adjusted Close  prices.')
    st.subheader('5.Comparing the forecasting accuracy using   2 different time intervals i.e., Daily and 15 minute intervals.')
               

# Section: Analysis

elif section == 'Analysis':
    st.header('Analysis')
    st.write('This is the analysis section.')
    st.subheader('General Plot Of Stocks')
    time_interval = st.selectbox("Time Interval",options = time_intervals)
    if time_interval == '15 MINUTES' :
        start_date = period 
        end_date = interval 
    elif time_interval == 'DAILY' :     
        start_date  = st.date_input('Start',value=pd.to_datetime('2016-01-01'))
        end_date  = st.date_input('End',value=pd.to_datetime('today'))
    dropdown = st.multiselect('Pick your assets',tickers)
    if len(dropdown)>0:
        #df=yf.download(dropdown,start,end)['Adj Close']
        st.header('Returns of {}'.format(dropdown))
        if time_interval == '15 MINUTES' :
            df=retrn(yf.download(dropdown,period='60d',interval='15m')['Adj Close'])
        elif time_interval == 'DAILY' :
            df=retrn(yf.download(dropdown,start_date ,end_date )['Adj Close'])
        st.line_chart(df)
        
        
        
        
     # Additional select boxes
    st.write('Please select additional options:')
    col1, col2, col3,col4 = st.columns(4)

    with col1:
        time_interval2 = st.selectbox("Duration", options=time_intervalss)
    with col2:
        stocks = st.selectbox("Stocks", tickers)
    with col3:
        price= st.selectbox("Price", ["Adj Close","Close"])    
    with col4:
        models = st.selectbox("Models", ["CNN-LSTM","GRU"])
    
  
    st.subheader('Model Training process')
    end =  date.today().strftime("%Y-%m-%d")
    start = '2016-01-01'
    
#1         
    if time_interval2 == "Daily" and stocks =="HDFCBANK.NS" and price == "Adj Close" and models == "CNN-LSTM":
        M=load_model("MODELS/cnn_hdfc_adj(D).h5")
        DF=yf.download(stocks,start,end)
        fig,valid,MAE,MAPE,MDAPE,final= plot_model(M, DF, stocks, price)
        st.pyplot(fig)
        #st.line_chart(valid)
        st.subheader('Residuals obtained for Test data')
        num_rows = st.slider("Number of last rows to display from the table", min_value=1, max_value=len(valid), value=10)
        selected_rows = valid.tail(num_rows)
        st.dataframe(selected_rows)
        summary = []
        M.summary(print_fn=lambda x: summary.append(x))
        # Convert the summary to a DataFrame
        df_summary = pd.read_csv(io.StringIO('\n'.join(summary[1:])), sep=r'\s{2,}', engine='python')
        st.subheader('Model Parameters')
        st.dataframe(df_summary)
        
#2   
    if time_interval2 == "Daily" and stocks =="TCS.NS" and price == "Adj Close" and models == "CNN-LSTM":
          M=load_model("MODELS/cnn_tcs_adj(D).h5")
          DF=yf.download(stocks,start,end)
          fig ,valid,MAE,MAPE,MDAPE,final= plot_model(M, DF, stocks, price)
          st.pyplot(fig) 
          st.subheader('Residuals obtained for Test data')
          num_rows = st.slider("Number of last rows to display from the table", min_value=1, max_value=len(valid), value=10)
          selected_rows = valid.tail(num_rows)
          st.dataframe(selected_rows)
          summary = []
          M.summary(print_fn=lambda x: summary.append(x))
          # Convert the summary to a DataFrame
          df_summary = pd.read_csv(io.StringIO('\n'.join(summary[1:])), sep=r'\s{2,}', engine='python')
          st.subheader('Model Parameters')
          st.dataframe(df_summary)
#3
    if time_interval2 == "Daily" and stocks =="POWERGRID.NS" and price == "Adj Close" and models == "CNN-LSTM":
          M=load_model("MODELS/cnn_powergrid_adj(D).h5")
          DF=yf.download(stocks,start,end)
          fig ,valid,MAE,MAPE,MDAPE ,final= plot_model(M, DF, stocks, price)
          st.pyplot(fig) 
          st.subheader('Residuals obtained for Test data')
          num_rows = st.slider("Number of last rows to display from the table", min_value=1, max_value=len(valid), value=10)
          selected_rows = valid.tail(num_rows)
          st.dataframe(selected_rows)
          summary = []
          M.summary(print_fn=lambda x: summary.append(x))
          # Convert the summary to a DataFrame
          df_summary = pd.read_csv(io.StringIO('\n'.join(summary[1:])), sep=r'\s{2,}', engine='python')
          st.subheader('Model Parameters')
          st.dataframe(df_summary)
 #4
    if time_interval2 == "Daily" and stocks =="BAJAJ-AUTO.NS" and price == "Adj Close" and models == "CNN-LSTM":
          M=load_model("MODELS/cnn_bajaj_adj(D).h5")
          DF=yf.download(stocks,start,end)
          fig,valid,MAE,MAPE,MDAPE ,final = plot_model(M, DF, stocks, price)
          st.pyplot(fig) 
          st.subheader('Residuals obtained for Test data')
          num_rows = st.slider("Number of last rows to display from the table", min_value=1, max_value=len(valid), value=10)
          selected_rows = valid.tail(num_rows)
          st.dataframe(selected_rows)
          summary = []
          M.summary(print_fn=lambda x: summary.append(x))
          # Convert the summary to a DataFrame
          df_summary = pd.read_csv(io.StringIO('\n'.join(summary[1:])), sep=r'\s{2,}', engine='python')
          st.subheader('Model Parameters')
          st.dataframe(df_summary)
          

#5         
    if time_interval2 == "Daily" and stocks =="HDFCBANK.NS" and price == "Close" and models == "CNN-LSTM":
        M=load_model("MODELS/cnn_hdfc_clo(D).h5")
        DF=yf.download(stocks,start,end)
        fig ,valid ,MAE,MAPE,MDAPE,final= plot_model(M, DF, stocks, price)
        st.pyplot(fig) 
        st.subheader('Residuals obtained for Test data')
        num_rows = st.slider("Number of last rows to display from the table", min_value=1, max_value=len(valid), value=10)
        selected_rows = valid.tail(num_rows)
        st.dataframe(selected_rows)
        summary = []
        M.summary(print_fn=lambda x: summary.append(x))
        # Convert the summary to a DataFrame
        df_summary = pd.read_csv(io.StringIO('\n'.join(summary[1:])), sep=r'\s{2,}', engine='python')
        st.subheader('Model Parameters')
        st.dataframe(df_summary)
#6  
    if time_interval2 == "Daily" and stocks =="TCS.NS" and price == "Close" and models == "CNN-LSTM":
          M=load_model("MODELS/cnn_tcs_clo(D).h5")
          DF=yf.download(stocks,start,end)
          fig ,valid,MAE,MAPE,MDAPE,final = plot_model(M, DF, stocks, price)
          st.pyplot(fig)
          st.subheader('Residuals obtained for Test data')
          num_rows = st.slider("Number of last rows to display from the table", min_value=1, max_value=len(valid), value=10)
          selected_rows = valid.tail(num_rows)
          st.dataframe(selected_rows)
          summary = []
          M.summary(print_fn=lambda x: summary.append(x))
          # Convert the summary to a DataFrame
          df_summary = pd.read_csv(io.StringIO('\n'.join(summary[1:])), sep=r'\s{2,}', engine='python')
          st.subheader('Model Parameters')
          st.dataframe(df_summary)
#7
    if time_interval2 == "Daily" and stocks =="POWERGRID.NS" and price == "Close" and models == "CNN-LSTM":
          M=load_model("MODELS/cnn_powergrid_clo(D).h5")
          DF=yf.download(stocks,start,end)
          fig,valid ,MAE,MAPE,MDAPE,final = plot_model(M, DF, stocks, price)
          st.pyplot(fig) 
          st.subheader('Residuals obtained for Test data')
          num_rows = st.slider("Number of last rows to display from the table", min_value=1, max_value=len(valid), value=10)
          selected_rows = valid.tail(num_rows)
          st.dataframe(selected_rows)
          summary = []
          M.summary(print_fn=lambda x: summary.append(x))
          # Convert the summary to a DataFrame
          df_summary = pd.read_csv(io.StringIO('\n'.join(summary[1:])), sep=r'\s{2,}', engine='python')
          st.subheader('Model Parameters')
          st.dataframe(df_summary)
 #8
    if time_interval2 == "Daily" and stocks =="BAJAJ-AUTO.NS" and price == "Close" and models == "CNN-LSTM":
          M=load_model("MODELS/cnn_bajaj_clo(D).h5")
          DF=yf.download(stocks,start,end)
          fig ,valid,MAE,MAPE,MDAPE,final = plot_model(M, DF, stocks, price)
          st.pyplot(fig)
          st.subheader('Residuals obtained for Test data')
          num_rows = st.slider("Number of last rows to display from the table", min_value=1, max_value=len(valid), value=10)
          selected_rows = valid.tail(num_rows)
          st.dataframe(selected_rows)
          summary = []
          M.summary(print_fn=lambda x: summary.append(x))
          # Convert the summary to a DataFrame
          df_summary = pd.read_csv(io.StringIO('\n'.join(summary[1:])), sep=r'\s{2,}', engine='python')
          st.subheader('Model Parameters')
          st.dataframe(df_summary)

#9         
    if time_interval2 == "15 minutes" and stocks =="HDFCBANK.NS" and price == "Adj Close" and models == "CNN-LSTM":
        M=load_model("MODELS/cnn_hdfc_adj(15m).h5")
        DF=yf.download(stocks,period='60d',interval='15m')
        fig ,valid,MAE,MAPE,MDAPE,final = plot_model(M, DF, stocks, price)
        st.pyplot(fig)
        st.subheader('Residuals obtained for Test data')
        num_rows = st.slider("Number of last rows to display from the table", min_value=1, max_value=len(valid), value=10)
        selected_rows = valid.tail(num_rows)
        st.dataframe(selected_rows)
        summary = []
        M.summary(print_fn=lambda x: summary.append(x))
        # Convert the summary to a DataFrame
        df_summary = pd.read_csv(io.StringIO('\n'.join(summary[1:])), sep=r'\s{2,}', engine='python')
        st.subheader('Model Parameters')
        st.dataframe(df_summary)
#10   
    if time_interval2 == "15 minutes" and stocks =="TCS.NS" and price == "Adj Close" and models == "CNN-LSTM":
          M=load_model("MODELS/cnn_tcs_adj(15m).h5")
          DF=yf.download(stocks,period='60d',interval='15m')
          fig,valid,MAE,MAPE,MDAPE,final = plot_model(M, DF, stocks, price)
          st.pyplot(fig)
          st.subheader('Residuals obtained for Test data')
          num_rows = st.slider("Number of last rows to display from the table", min_value=1, max_value=len(valid), value=10)
          selected_rows = valid.tail(num_rows)
          st.dataframe(selected_rows)
          summary = []
          M.summary(print_fn=lambda x: summary.append(x))
          # Convert the summary to a DataFrame
          df_summary = pd.read_csv(io.StringIO('\n'.join(summary[1:])), sep=r'\s{2,}', engine='python')
          st.subheader('Model Parameters')
          st.dataframe(df_summary)
#11
    if time_interval2 == "15 minutes" and stocks =="POWERGRID.NS" and price == "Adj Close" and models == "CNN-LSTM":
          M=load_model("MODELS/cnn_powergrid_adj(15m).h5")
          DF=yf.download(stocks,period='60d',interval='15m')
          fig ,valid,MAE,MAPE,MDAPE,final = plot_model(M, DF, stocks, price)
          st.pyplot(fig) 
          st.subheader('Residuals obtained for Test data')
          num_rows = st.slider("Number of last rows to display from the table", min_value=1, max_value=len(valid), value=10)
          selected_rows = valid.tail(num_rows)
          st.dataframe(selected_rows)
          summary = []
          M.summary(print_fn=lambda x: summary.append(x))
          # Convert the summary to a DataFrame
          df_summary = pd.read_csv(io.StringIO('\n'.join(summary[1:])), sep=r'\s{2,}', engine='python')
          st.subheader('Model Parameters')
          st.dataframe(df_summary)
 #12
    if time_interval2 == "15 minutes" and stocks =="BAJAJ-AUTO.NS" and price == "Adj Close" and models == "CNN-LSTM":
          M=load_model("MODELS/cnn_bajaj_adj(15m).h5")
          DF=yf.download(stocks,period='60d',interval='15m')
          fig ,valid,MAE,MAPE,MDAPE ,final= plot_model(M, DF, stocks, price)
          st.pyplot(fig) 
          st.subheader('Residuals obtained for Test data')
          num_rows = st.slider("Number of last rows to display from the table", min_value=1, max_value=len(valid), value=10)
          selected_rows = valid.tail(num_rows)
          st.dataframe(selected_rows)
          summary = []
          M.summary(print_fn=lambda x: summary.append(x))
          # Convert the summary to a DataFrame
          df_summary = pd.read_csv(io.StringIO('\n'.join(summary[1:])), sep=r'\s{2,}', engine='python')
          st.subheader('Model Parameters')
          st.dataframe(df_summary)
#GRU
#1         
    if time_interval2 == "Daily" and stocks =="HDFCBANK.NS" and price == "Adj Close" and models == "GRU":
        M=load_model("MODELS/GRU_hdfc_adj(D).h5")
        DF=yf.download(stocks,start,end)
        fig,valid,MAE,MAPE,MDAPE,final  = plot_model(M, DF, stocks, price)
        st.pyplot(fig)
        st.subheader('Residuals obtained for Test data')
        num_rows = st.slider("Number of last rows to display from the table", min_value=1, max_value=len(valid), value=10)
        selected_rows = valid.tail(num_rows)
        st.dataframe(selected_rows)
        summary = []
        M.summary(print_fn=lambda x: summary.append(x))
        # Convert the summary to a DataFrame
        df_summary = pd.read_csv(io.StringIO('\n'.join(summary[1:])), sep=r'\s{2,}', engine='python')
        st.subheader('Model Parameters')
        st.dataframe(df_summary)
#2   
    if time_interval2 == "Daily" and stocks =="TCS.NS" and price == "Adj Close" and models == "GRU":
          M=load_model("MODELS/GRU_tcs_adj(D).h5")
          DF=yf.download(stocks,start,end)
          fig,valid,MAE,MAPE,MDAPE,final  = plot_model(M, DF, stocks, price)
          st.pyplot(fig)  
          st.subheader('Residuals obtained for Test data')
          num_rows = st.slider("Number of last rows to display from the table", min_value=1, max_value=len(valid), value=10)
          selected_rows = valid.tail(num_rows)
          st.dataframe(selected_rows)
          summary = []
          M.summary(print_fn=lambda x: summary.append(x))
          # Convert the summary to a DataFrame
          df_summary = pd.read_csv(io.StringIO('\n'.join(summary[1:])), sep=r'\s{2,}', engine='python')
          st.subheader('Model Parameters')
          st.dataframe(df_summary)
#3
    if time_interval2 == "Daily" and stocks =="POWERGRID.NS" and price == "Adj Close" and models == "GRU":
          M=load_model("MODELS/GRU_powergrid_adj(D).h5")
          DF=yf.download(stocks,start,end)
          fig ,valid,MAE,MAPE,MDAPE,final = plot_model(M, DF, stocks, price)
          st.pyplot(fig) 
          st.subheader('Residuals obtained for Test data')
          num_rows = st.slider("Number of last rows to display from the table", min_value=1, max_value=len(valid), value=10)
          selected_rows = valid.tail(num_rows)
          st.dataframe(selected_rows)
          summary = []
          M.summary(print_fn=lambda x: summary.append(x))
          # Convert the summary to a DataFrame
          df_summary = pd.read_csv(io.StringIO('\n'.join(summary[1:])), sep=r'\s{2,}', engine='python')
          st.subheader('Model Parameters')
          st.dataframe(df_summary)
 #4
    if time_interval2 == "Daily" and stocks =="BAJAJ-AUTO.NS" and price == "Adj Close" and models == "GRU":
          M=load_model("MODELS/GRU_bajaj_adj(D).h5")
          DF=yf.download(stocks,start,end)
          fig ,valid,MAE,MAPE,MDAPE,final = plot_model(M, DF, stocks, price)
          st.pyplot(fig)
          st.subheader('Residuals obtained for Test data')
          num_rows = st.slider("Number of last rows to display from the table", min_value=1, max_value=len(valid), value=10)
          selected_rows = valid.tail(num_rows)
          st.dataframe(selected_rows)
          summary = []
          M.summary(print_fn=lambda x: summary.append(x))
          # Convert the summary to a DataFrame
          df_summary = pd.read_csv(io.StringIO('\n'.join(summary[1:])), sep=r'\s{2,}', engine='python')
          st.subheader('Model Parameters')
          st.dataframe(df_summary)
           

#5         
    if time_interval2 == "Daily" and stocks =="HDFCBANK.NS" and price == "Close" and models == "GRU":
        M=load_model("MODELS/GRU_hdfc_clo(D).h5")
        DF=yf.download(stocks,start,end)
        fig,valid,MAE,MAPE,MDAPE ,final = plot_model(M, DF, stocks, price)
        st.pyplot(fig)
        st.subheader('Residuals obtained for Test data')
        num_rows = st.slider("Number of last rows to display from the table", min_value=1, max_value=len(valid), value=10)
        selected_rows = valid.tail(num_rows)
        st.dataframe(selected_rows)
        summary = []
        M.summary(print_fn=lambda x: summary.append(x))
        # Convert the summary to a DataFrame
        df_summary = pd.read_csv(io.StringIO('\n'.join(summary[1:])), sep=r'\s{2,}', engine='python')
        st.subheader('Model Parameters')
        st.dataframe(df_summary)
#6  
    if time_interval2 == "Daily" and stocks =="TCS.NS" and price == "Close" and models == "GRU":
          M=load_model("MODELS/GRU_tcs_clo(D).h5")
          DF=yf.download(stocks,start,end)
          fig,valid,MAE,MAPE,MDAPE,final  = plot_model(M, DF, stocks, price)
          st.pyplot(fig) 
          st.subheader('Residuals obtained for Test data')
          num_rows = st.slider("Number of last rows to display from the table", min_value=1, max_value=len(valid), value=10)
          selected_rows = valid.tail(num_rows)
          st.dataframe(selected_rows)
          summary = []
          M.summary(print_fn=lambda x: summary.append(x))
          # Convert the summary to a DataFrame
          df_summary = pd.read_csv(io.StringIO('\n'.join(summary[1:])), sep=r'\s{2,}', engine='python')
          st.subheader('Model Parameters')
          st.dataframe(df_summary)
#7
    if time_interval2 == "Daily" and stocks =="POWERGRID.NS" and price == "Close" and models == "GRU":
          M=load_model("MODELS/GRU_powergrid_clo(D).h5")
          DF=yf.download(stocks,start,end)
          fig,valid ,MAE,MAPE,MDAPE,final = plot_model(M, DF, stocks, price)
          st.pyplot(fig)
          st.subheader('Residuals obtained for Test data')
          num_rows = st.slider("Number of last rows to display from the table", min_value=1, max_value=len(valid), value=10)
          selected_rows = valid.tail(num_rows)
          st.dataframe(selected_rows)
          summary = []
          M.summary(print_fn=lambda x: summary.append(x))
          # Convert the summary to a DataFrame
          df_summary = pd.read_csv(io.StringIO('\n'.join(summary[1:])), sep=r'\s{2,}', engine='python')
          st.subheader('Model Parameters')
          st.dataframe(df_summary)
 #8
    if time_interval2 == "Daily" and stocks =="BAJAJ-AUTO.NS" and price == "Close" and models == "GRU":
          M=load_model("MODELS/GRU_bajaj_clo(D).h5")
          DF=yf.download(stocks,start,end)
          fig,valid,MAE,MAPE,MDAPE ,final = plot_model(M, DF, stocks, price)
          st.pyplot(fig)
          st.subheader('Residuals obtained for Test data')
          num_rows = st.slider("Number of last rows to display from the table", min_value=1, max_value=len(valid), value=10)
          selected_rows = valid.tail(num_rows)
          st.dataframe(selected_rows)
          summary = []
          M.summary(print_fn=lambda x: summary.append(x))
          # Convert the summary to a DataFrame
          df_summary = pd.read_csv(io.StringIO('\n'.join(summary[1:])), sep=r'\s{2,}', engine='python')
          st.subheader('Model Parameters')
          st.dataframe(df_summary)

#9         
    if time_interval2 == "15 minutes" and stocks =="HDFCBANK.NS" and price == "Adj Close" and models == "GRU":
        M=load_model("MODELS/GRU_hdfc_adj(15m).h5")
        DF=yf.download(stocks,period='60d',interval='15m')
        fig,valid,MAE,MAPE,MDAPE,final  = plot_model(M, DF, stocks, price)
        st.pyplot(fig) 
        st.subheader('Residuals obtained for Test data')
        num_rows = st.slider("Number of last rows to display from the table", min_value=1, max_value=len(valid), value=10)
        selected_rows = valid.tail(num_rows)
        st.dataframe(selected_rows)
        summary = []
        M.summary(print_fn=lambda x: summary.append(x))
        # Convert the summary to a DataFrame
        df_summary = pd.read_csv(io.StringIO('\n'.join(summary[1:])), sep=r'\s{2,}', engine='python')
        st.subheader('Model Parameters')
        st.dataframe(df_summary)
#10   
    if time_interval2 == "15 minutes" and stocks =="TCS.NS" and price == "Adj Close" and models == "GRU":
          M=load_model("MODELS/GRU_tcs_adj(15m).h5")
          DF=yf.download(stocks,period='60d',interval='15m')
          fig,valid ,MAE,MAPE,MDAPE ,final= plot_model(M, DF, stocks, price)
          st.pyplot(fig) 
          st.subheader('Residuals obtained for Test data')
          num_rows = st.slider("Number of last rows to display from the table", min_value=1, max_value=len(valid), value=10)
          selected_rows = valid.tail(num_rows)
          st.dataframe(selected_rows)
          summary = []
          M.summary(print_fn=lambda x: summary.append(x))
          # Convert the summary to a DataFrame
          df_summary = pd.read_csv(io.StringIO('\n'.join(summary[1:])), sep=r'\s{2,}', engine='python')
          st.subheader('Model Parameters')
          st.dataframe(df_summary)
#11
    if time_interval2 == "15 minutes" and stocks =="POWERGRID.NS" and price == "Adj Close" and models == "GRU":
          M=load_model("MODELS/GRU_powergrid_adj(15m).h5")
          DF=yf.download(stocks,period='60d',interval='15m')
          fig ,valid ,MAE,MAPE,MDAPE,final= plot_model(M, DF, stocks, price)
          st.pyplot(fig) 
          st.subheader('Residuals obtained for Test data')
          num_rows = st.slider("Number of last rows to display from the table", min_value=1, max_value=len(valid), value=10)
          selected_rows = valid.tail(num_rows)
          st.dataframe(selected_rows)
          summary = []
          M.summary(print_fn=lambda x: summary.append(x))
          # Convert the summary to a DataFrame
          df_summary = pd.read_csv(io.StringIO('\n'.join(summary[1:])), sep=r'\s{2,}', engine='python')
          st.subheader('Model Parameters')
          st.dataframe(df_summary)
 #12
    if time_interval2 == "15 minutes" and stocks =="BAJAJ-AUTO.NS" and price == "Adj Close" and models == "GRU":
          M=load_model("MODELS/GRU_bajaj_adj(15m).h5")
          DF=yf.download(stocks,period='60d',interval='15m')
          fig ,valid,MAE,MAPE,MDAPE ,final= plot_model(M, DF, stocks, price)
          st.pyplot(fig)  
          st.subheader('Residuals obtained for Test data')
          num_rows = st.slider("Number of last rows to display from the table", min_value=1, max_value=len(valid), value=10)
          selected_rows = valid.tail(num_rows)
          st.dataframe(selected_rows)
          summary = []
          M.summary(print_fn=lambda x: summary.append(x))
          # Convert the summary to a DataFrame
          df_summary = pd.read_csv(io.StringIO('\n'.join(summary[1:])), sep=r'\s{2,}', engine='python')
          st.subheader('Model Parameters')
          st.dataframe(df_summary)


# Section: Prediction
elif section == 'Prediction':
    st.header('Prediction')
    st.subheader('This is the prediction section')
    col5, col6, col7,col8 = st.columns(4)

    with col5:
        time_interval2 = st.selectbox("Duration", options=time_intervalss)
    with col6:
        stocks = st.selectbox("Stocks", tickers)
    with col7:
        price= st.selectbox("Price", ["Adj Close","Close"])    
    with col8:
        models = st.selectbox("Models", ["CNN-LSTM","GRU"])
        
        end =  date.today().strftime("%Y-%m-%d")
        start = '2016-01-01'
 #1       
    if time_interval2 == "Daily" and stocks =="HDFCBANK.NS" and price == "Adj Close" and models == "CNN-LSTM":
        M=load_model("MODELS/cnn_hdfc_adj(D).h5")
        DF=yf.download(stocks,start,end)
        fig,valid,MAE,MAPE,MDAPE,final= plot_model(M, DF, stocks, price)
        st.subheader(f'Predictions from {models} Model for next 5 days')
        if __name__ == "__main__":
            present_time = DF.index[-1].to_pydatetime()  # Replace with your specific present time
            next_date = get_date(present_time)
        next_date=np.array(next_date).reshape(5,1)  
        final.index = list(next_date.flatten())
        final.index.name = "Date"
        st.dataframe(final)
        st.subheader('Model Error Rate')
        st.write("Mean Absolute Error (MAE) =",MAE)
        st.write("Mean Absolute Percentage Error (MAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MAPE), unsafe_allow_html=True)
        st.write("Median Absolute Percentage Error (MDAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MDAPE), unsafe_allow_html=True)
        

        
#2   
    if time_interval2 == "Daily" and stocks =="TCS.NS" and price == "Adj Close" and models == "CNN-LSTM":
          M=load_model("MODELS/cnn_tcs_adj(D).h5")
          DF=yf.download(stocks,start,end)
          fig ,valid,MAE,MAPE,MDAPE,final= plot_model(M, DF, stocks, price)
          st.subheader(f'Predictions from {models} Model for next 5 days')
          if __name__ == "__main__":
              present_time = DF.index[-1].to_pydatetime()  # Replace with your specific present time
              next_date = get_date(present_time)
          next_date=np.array(next_date).reshape(5,1)  
          final.index = list(next_date.flatten())
          final.index.name = "Date"
          st.dataframe(final)
          st.subheader('Model Error Rate')
          st.write("Mean Absolute Error (MAE) =",MAE)
          st.write("Mean Absolute Percentage Error (MAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MAPE), unsafe_allow_html=True)
          st.write("Median Absolute Percentage Error (MDAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MDAPE), unsafe_allow_html=True)

#3
    if time_interval2 == "Daily" and stocks =="POWERGRID.NS" and price == "Adj Close" and models == "CNN-LSTM":
          M=load_model("MODELS/cnn_powergrid_adj(D).h5")
          DF=yf.download(stocks,start,end)
          fig ,valid,MAE,MAPE,MDAPE,final = plot_model(M, DF, stocks, price)
          st.subheader(f'Predictions from {models} Model for next 5 days')
          if __name__ == "__main__":
              present_time = DF.index[-1].to_pydatetime()  # Replace with your specific present time
              next_date = get_date(present_time)
          next_date=np.array(next_date).reshape(5,1)  
          final.index = list(next_date.flatten())
          final.index.name = "Date"
          st.dataframe(final)
          st.subheader('Model Error Rate')
          st.write("Mean Absolute Error (MAE) =",MAE)
          st.write("Mean Absolute Percentage Error (MAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MAPE), unsafe_allow_html=True)
          st.write("Median Absolute Percentage Error (MDAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MDAPE), unsafe_allow_html=True)

 #4
    if time_interval2 == "Daily" and stocks =="BAJAJ-AUTO.NS" and price == "Adj Close" and models == "CNN-LSTM":
          M=load_model("MODELS/cnn_bajaj_adj(D).h5")
          DF=yf.download(stocks,start,end)
          fig,valid,MAE,MAPE,MDAPE ,final = plot_model(M, DF, stocks, price)
          st.subheader(f'Predictions from {models} Model for next 5 days')
          if __name__ == "__main__":
              present_time = DF.index[-1].to_pydatetime()  # Replace with your specific present time
              next_date = get_date(present_time)
          next_date=np.array(next_date).reshape(5,1)  
          final.index = list(next_date.flatten())
          final.index.name = "Date"
          st.dataframe(final)
          st.subheader('Model Error Rate')
          st.write("Mean Absolute Error (MAE) =",MAE)
          st.write("Mean Absolute Percentage Error (MAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MAPE), unsafe_allow_html=True)
          st.write("Median Absolute Percentage Error (MDAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MDAPE), unsafe_allow_html=True)

          

#5         
    if time_interval2 == "Daily" and stocks =="HDFCBANK.NS" and price == "Close" and models == "CNN-LSTM":
        M=load_model("MODELS/cnn_hdfc_clo(D).h5")
        DF=yf.download(stocks,start,end)
        fig ,valid ,MAE,MAPE,MDAPE,final= plot_model(M, DF, stocks, price)
        st.subheader(f'Predictions from {models} Model for next 5 days')
        if __name__ == "__main__":
            present_time = DF.index[-1].to_pydatetime()  # Replace with your specific present time
            next_date = get_date(present_time)
        next_date=np.array(next_date).reshape(5,1)  
        final.index = list(next_date.flatten())
        final.index.name = "Date"
        st.dataframe(final)
        st.subheader('Model Error Rate')
        st.write("Mean Absolute Error (MAE) =",MAE)
        st.write("Mean Absolute Percentage Error (MAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MAPE), unsafe_allow_html=True)
        st.write("Median Absolute Percentage Error (MDAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MDAPE), unsafe_allow_html=True)

#6  
    if time_interval2 == "Daily" and stocks =="TCS.NS" and price == "Close" and models == "CNN-LSTM":
          M=load_model("MODELS/cnn_tcs_clo(D).h5")
          DF=yf.download(stocks,start,end)
          fig ,valid,MAE,MAPE,MDAPE ,final= plot_model(M, DF, stocks, price)
          st.subheader(f'Predictions from {models} Model for next 5 days')
          if __name__ == "__main__":
              present_time = DF.index[-1].to_pydatetime()  # Replace with your specific present time
              next_date = get_date(present_time)
          next_date=np.array(next_date).reshape(5,1)  
          final.index = list(next_date.flatten())
          final.index.name = "Date"
          st.dataframe(final)
          st.subheader('Model Error Rate')
          st.write("Mean Absolute Error (MAE) =",MAE)
          st.write("Mean Absolute Percentage Error (MAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MAPE), unsafe_allow_html=True)
          st.write("Median Absolute Percentage Error (MDAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MDAPE), unsafe_allow_html=True)

#7
    if time_interval2 == "Daily" and stocks =="POWERGRID.NS" and price == "Close" and models == "CNN-LSTM":
          M=load_model("MODELS/cnn_powergrid_clo(D).h5")
          DF=yf.download(stocks,start,end)
          fig,valid ,MAE,MAPE,MDAPE,final = plot_model(M, DF, stocks, price)
          st.subheader(f'Predictions from {models} Model for next 5 days')
          if __name__ == "__main__":
              present_time = DF.index[-1].to_pydatetime()  # Replace with your specific present time
              next_date = get_date(present_time)
          next_date=np.array(next_date).reshape(5,1)  
          final.index = list(next_date.flatten())
          final.index.name = "Date"
          st.dataframe(final)
          st.subheader('Model Error Rate')
          st.write("Mean Absolute Error (MAE) =",MAE)
          st.write("Mean Absolute Percentage Error (MAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MAPE), unsafe_allow_html=True)
          st.write("Median Absolute Percentage Error (MDAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MDAPE), unsafe_allow_html=True)

 #8
    if time_interval2 == "Daily" and stocks =="BAJAJ-AUTO.NS" and price == "Close" and models == "CNN-LSTM":
          M=load_model("MODELS/cnn_bajaj_clo(D).h5")
          DF=yf.download(stocks,start,end)
          fig ,valid,MAE,MAPE,MDAPE,final = plot_model(M, DF, stocks, price)
          st.subheader(f'Predictions from {models} Model for next 5 days')
          if __name__ == "__main__":
              present_time = DF.index[-1].to_pydatetime()  # Replace with your specific present time
              next_date = get_date(present_time)
          next_date=np.array(next_date).reshape(5,1)  
          final.index = list(next_date.flatten())
          final.index.name = "Date"
          st.dataframe(final)
          st.subheader('Model Error Rate')
          st.write("Mean Absolute Error (MAE) =",MAE)
          st.write("Mean Absolute Percentage Error (MAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MAPE), unsafe_allow_html=True)
          st.write("Median Absolute Percentage Error (MDAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MDAPE), unsafe_allow_html=True)


#9         
    if time_interval2 == "15 minutes" and stocks =="HDFCBANK.NS" and price == "Adj Close" and models == "CNN-LSTM":
        M=load_model("MODELS/cnn_hdfc_adj(15m).h5")
        DF=yf.download(stocks,period='60d',interval='15m')
        fig ,valid,MAE,MAPE,MDAPE,final = plot_model(M, DF, stocks, price)
        st.subheader(f'Predictions from {models} Model')
        if __name__ == "__main__":
            last_time_point = DF.index[-1].to_pydatetime()  # Replace with your specific last time index
            next_time = get_time(last_time_point)
        next_time=np.array(next_time).reshape(5,1)  
        final.index = list(next_time.flatten())
        final.index.name = "DateTime"
        st.dataframe(final)
        st.subheader('Model Error Rate')
        st.write("Mean Absolute Error (MAE) =",MAE)
        st.write("Mean Absolute Percentage Error (MAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MAPE), unsafe_allow_html=True)
        st.write("Median Absolute Percentage Error (MDAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MDAPE), unsafe_allow_html=True)

#10   
    if time_interval2 == "15 minutes" and stocks =="TCS.NS" and price == "Adj Close" and models == "CNN-LSTM":
          M=load_model("MODELS/cnn_tcs_adj(15m).h5")
          DF=yf.download(stocks,period='60d',interval='15m')
          fig,valid,MAE,MAPE,MDAPE,final  = plot_model(M, DF, stocks, price)
          st.subheader(f'Predictions from {models} Model')
          if __name__ == "__main__":
              last_time_point = DF.index[-1].to_pydatetime()  # Replace with your specific last time index
              next_time = get_time(last_time_point)
          next_time=np.array(next_time).reshape(5,1)  
          final.index = list(next_time.flatten())
          final.index.name = "DateTime"
          st.dataframe(final)
          st.subheader('Model Error Rate')
          st.write("Mean Absolute Error (MAE) =",MAE)
          st.write("Mean Absolute Percentage Error (MAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MAPE), unsafe_allow_html=True)
          st.write("Median Absolute Percentage Error (MDAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MDAPE), unsafe_allow_html=True)

#11
    if time_interval2 == "15 minutes" and stocks =="POWERGRID.NS" and price == "Adj Close" and models == "CNN-LSTM":
          M=load_model("MODELS/cnn_powergrid_adj(15m).h5")
          DF=yf.download(stocks,period='60d',interval='15m')
          fig ,valid,MAE,MAPE,MDAPE,final = plot_model(M, DF, stocks, price)
          st.subheader(f'Predictions from {models} Model')
          if __name__ == "__main__":
              last_time_point = DF.index[-1].to_pydatetime()  # Replace with your specific last time index
              next_time = get_time(last_time_point)
          next_time=np.array(next_time).reshape(5,1)  
          final.index = list(next_time.flatten())
          final.index.name = "DateTime"
          st.dataframe(final)
          st.subheader('Model Error Rate')
          st.write("Mean Absolute Error (MAE) =",MAE)
          st.write("Mean Absolute Percentage Error (MAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MAPE), unsafe_allow_html=True)
          st.write("Median Absolute Percentage Error (MDAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MDAPE), unsafe_allow_html=True)

 #12
    if time_interval2 == "15 minutes" and stocks =="BAJAJ-AUTO.NS" and price == "Adj Close" and models == "CNN-LSTM":
          M=load_model("MODELS/cnn_bajaj_adj(15m).h5")
          DF=yf.download(stocks,period='60d',interval='15m')
          fig ,valid,MAE,MAPE,MDAPE,final = plot_model(M, DF, stocks, price)
          st.subheader(f'Predictions from {models} Model')
          if __name__ == "__main__":
              last_time_point = DF.index[-1].to_pydatetime()  # Replace with your specific last time index
              next_time = get_time(last_time_point)
          next_time=np.array(next_time).reshape(5,1)  
          final.index = list(next_time.flatten())
          final.index.name = "DateTime"
          st.dataframe(final)
          st.subheader('Model Error Rate')
          st.write("Mean Absolute Error (MAE) =",MAE)
          st.write("Mean Absolute Percentage Error (MAPE)=",MAPE)
          st.write("Median Absolute Percentage Error (MDAPE) =",MDAPE)

#GRU
#1         
    if time_interval2 == "Daily" and stocks =="HDFCBANK.NS" and price == "Adj Close" and models == "GRU":
        M=load_model("MODELS/GRU_hdfc_adj(D).h5")
        DF=yf.download(stocks,start,end)
        fig,valid,MAE,MAPE,MDAPE,final  = plot_model(M, DF, stocks, price)
        st.subheader(f'Predictions from {models} Model for next 5 days')
        if __name__ == "__main__":
            present_time = DF.index[-1].to_pydatetime()  # Replace with your specific present time
            next_date = get_date(present_time)
        next_date=np.array(next_date).reshape(5,1)  
        final.index = list(next_date.flatten())
        final.index.name = "Date"
        st.dataframe(final)
        st.subheader('Model Error Rate')
        st.write("Mean Absolute Error (MAE) =",MAE)
        st.write("Mean Absolute Percentage Error (MAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MAPE), unsafe_allow_html=True)
        st.write("Median Absolute Percentage Error (MDAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MDAPE), unsafe_allow_html=True)

#2   
    if time_interval2 == "Daily" and stocks =="TCS.NS" and price == "Adj Close" and models == "GRU":
          M=load_model("MODELS/GRU_tcs_adj(D).h5")
          DF=yf.download(stocks,start,end)
          fig,valid,MAE,MAPE,MDAPE,final  = plot_model(M, DF, stocks, price)
          st.subheader(f'Predictions from {models} Model for next 5 days')
          if __name__ == "__main__":
              present_time = DF.index[-1].to_pydatetime()  # Replace with your specific present time
              next_date = get_date(present_time)
          next_date=np.array(next_date).reshape(5,1)  
          final.index = list(next_date.flatten())
          final.index.name = "Date"
          st.dataframe(final)
          st.subheader('Model Error Rate')
          st.write("Mean Absolute Error (MAE) =",MAE)
          st.write("Mean Absolute Percentage Error (MAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MAPE), unsafe_allow_html=True)
          st.write("Median Absolute Percentage Error (MDAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MDAPE), unsafe_allow_html=True)

#3
    if time_interval2 == "Daily" and stocks =="POWERGRID.NS" and price == "Adj Close" and models == "GRU":
          M=load_model("MODELS/GRU_powergrid_adj(D).h5")
          DF=yf.download(stocks,start,end)
          fig ,valid,MAE,MAPE,MDAPE ,final= plot_model(M, DF, stocks, price)
          st.subheader(f'Predictions from {models} Model for next 5 days')
          if __name__ == "__main__":
              present_time = DF.index[-1].to_pydatetime()  # Replace with your specific present time
              next_date = get_date(present_time)
          next_date=np.array(next_date).reshape(5,1)  
          final.index = list(next_date.flatten())
          final.index.name = "Date"
          st.dataframe(final)
          st.subheader('Model Error Rate')
          st.write("Mean Absolute Error (MAE) =",MAE)
          st.write("Mean Absolute Percentage Error (MAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MAPE), unsafe_allow_html=True)
          st.write("Median Absolute Percentage Error (MDAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MDAPE), unsafe_allow_html=True)

 #4
    if time_interval2 == "Daily" and stocks =="BAJAJ-AUTO.NS" and price == "Adj Close" and models == "GRU":
          M=load_model("MODELS/GRU_bajaj_adj(D).h5")
          DF=yf.download(stocks,start,end)
          fig ,valid,MAE,MAPE,MDAPE,final = plot_model(M, DF, stocks, price)
          st.subheader(f'Predictions from {models} Model for next 5 days')
          if __name__ == "__main__":
              present_time = DF.index[-1].to_pydatetime()  # Replace with your specific present time
              next_date = get_date(present_time)
          next_date=np.array(next_date).reshape(5,1)  
          final.index = list(next_date.flatten())
          final.index.name = "Date"
          st.dataframe(final)
          st.subheader('Model Error Rate')
          st.write("Mean Absolute Error (MAE) =",MAE)
          st.write("Mean Absolute Percentage Error (MAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MAPE), unsafe_allow_html=True)
          st.write("Median Absolute Percentage Error (MDAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MDAPE), unsafe_allow_html=True)

           

#5         
    if time_interval2 == "Daily" and stocks =="HDFCBANK.NS" and price == "Close" and models == "GRU":
        M=load_model("MODELS/GRU_hdfc_clo(D).h5")
        DF=yf.download(stocks,start,end)
        fig,valid,MAE,MAPE,MDAPE ,final = plot_model(M, DF, stocks, price)
        st.subheader(f'Predictions from {models} Model for next 5 days')
        if __name__ == "__main__":
            present_time = DF.index[-1].to_pydatetime()  # Replace with your specific present time
            next_date = get_date(present_time)
        next_date=np.array(next_date).reshape(5,1)  
        final.index = list(next_date.flatten())
        final.index.name = "Date"
        st.dataframe(final)
        st.subheader('Model Error Rate')
        st.write("Mean Absolute Error (MAE) =",MAE)
        st.write("Mean Absolute Percentage Error (MAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MAPE), unsafe_allow_html=True)
        st.write("Median Absolute Percentage Error (MDAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MDAPE), unsafe_allow_html=True)

#6  
    if time_interval2 == "Daily" and stocks =="TCS.NS" and price == "Close" and models == "GRU":
          M=load_model("MODELS/GRU_tcs_clo(D).h5")
          DF=yf.download(stocks,start,end)
          fig,valid,MAE,MAPE,MDAPE ,final = plot_model(M, DF, stocks, price)
          st.subheader(f'Predictions from {models} Model for next 5 days')
          if __name__ == "__main__":
              present_time = DF.index[-1].to_pydatetime()  # Replace with your specific present time
              next_date = get_date(present_time)
          next_date=np.array(next_date).reshape(5,1)  
          final.index = list(next_date.flatten())
          final.index.name = "Date"
          st.dataframe(final)
          st.subheader('Model Error Rate')
          st.write("Mean Absolute Error (MAE) =",MAE)
          st.write("Mean Absolute Percentage Error (MAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MAPE), unsafe_allow_html=True)
          st.write("Median Absolute Percentage Error (MDAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MDAPE), unsafe_allow_html=True)

#7
    if time_interval2 == "Daily" and stocks =="POWERGRID.NS" and price == "Close" and models == "GRU":
          M=load_model("MODELS/GRU_powergrid_clo(D).h5")
          DF=yf.download(stocks,start,end)
          fig,valid ,MAE,MAPE,MDAPE ,final= plot_model(M, DF, stocks, price)
          st.subheader(f'Predictions from {models} Model for next 5 days')
          if __name__ == "__main__":
              present_time = DF.index[-1].to_pydatetime()  # Replace with your specific present time
              next_date = get_date(present_time)
          next_date=np.array(next_date).reshape(5,1)  
          final.index = list(next_date.flatten())
          final.index.name = "Date"
          st.dataframe(final)
          st.subheader('Model Error Rate')
          st.write("Mean Absolute Error (MAE) =",MAE)
          st.write("Mean Absolute Percentage Error (MAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MAPE), unsafe_allow_html=True)
          st.write("Median Absolute Percentage Error (MDAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MDAPE), unsafe_allow_html=True)

 #8
    if time_interval2 == "Daily" and stocks =="BAJAJ-AUTO.NS" and price == "Close" and models == "GRU":
          M=load_model("MODELS/GRU_bajaj_clo(D).h5")
          DF=yf.download(stocks,start,end)
          fig,valid,MAE,MAPE,MDAPE ,final = plot_model(M, DF, stocks, price)
          st.subheader(f'Predictions from {models} Model for next 5 days')
          if __name__ == "__main__":
              present_time = DF.index[-1].to_pydatetime()  # Replace with your specific present time
              next_date = get_date(present_time)
          next_date=np.array(next_date).reshape(5,1)  
          final.index = list(next_date.flatten())
          final.index.name = "Date"
          st.dataframe(final)
          st.subheader('Model Error Rate')
          st.write("Mean Absolute Error (MAE) =",MAE)
          st.write("Mean Absolute Percentage Error (MAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MAPE), unsafe_allow_html=True)
          st.write("Median Absolute Percentage Error (MDAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MDAPE), unsafe_allow_html=True)


#9         
    if time_interval2 == "15 minutes" and stocks =="HDFCBANK.NS" and price == "Adj Close" and models == "GRU":
        M=load_model("MODELS/GRU_hdfc_adj(15m).h5")
        DF=yf.download(stocks,period='60d',interval='15m')
        fig,valid,MAE,MAPE,MDAPE,final  = plot_model(M, DF, stocks, price)
        st.subheader(f'Predictions from {models} Model')
        if __name__ == "__main__":
            last_time_point = DF.index[-1].to_pydatetime()  # Replace with your specific last time index
            next_time = get_time(last_time_point)
        next_time=np.array(next_time).reshape(5,1)  
        final.index = list(next_time.flatten())
        final.index.name = "DateTime"
        st.dataframe(final)
        st.subheader('Model Error Rate')
        st.write("Mean Absolute Error (MAE) =",MAE)
        st.write("Mean Absolute Percentage Error (MAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MAPE), unsafe_allow_html=True)
        st.write("Median Absolute Percentage Error (MDAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MDAPE), unsafe_allow_html=True)

#10   
    if time_interval2 == "15 minutes" and stocks =="TCS.NS" and price == "Adj Close" and models == "GRU":
          M=load_model("MODELS/GRU_tcs_adj(15m).h5")
          DF=yf.download(stocks,period='60d',interval='15m')
          fig,valid ,MAE,MAPE,MDAPE ,final= plot_model(M, DF, stocks, price)
          st.subheader(f'Predictions from {models} Model')
          if __name__ == "__main__":
              last_time_point = DF.index[-1].to_pydatetime()  # Replace with your specific last time index
              next_time = get_time(last_time_point)
          next_time=np.array(next_time).reshape(5,1)  
          final.index = list(next_time.flatten())
          final.index.name = "DateTime"
          st.dataframe(final)
          st.subheader('Model Error Rate')
          st.write("Mean Absolute Error (MAE) =",MAE)
          st.write("Mean Absolute Percentage Error (MAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MAPE), unsafe_allow_html=True)
          st.write("Median Absolute Percentage Error (MDAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MDAPE), unsafe_allow_html=True)

#11
    if time_interval2 == "15 minutes" and stocks =="POWERGRID.NS" and price == "Adj Close" and models == "GRU":
          M=load_model("MODELS/GRU_powergrid_adj(15m).h5")
          DF=yf.download(stocks,period='60d',interval='15m')
          fig ,valid ,MAE,MAPE,MDAPE,final= plot_model(M, DF, stocks, price)
          st.subheader(f'Predictions from {models} Model')
          if __name__ == "__main__":
              last_time_point = DF.index[-1].to_pydatetime()  # Replace with your specific last time index
              next_time = get_time(last_time_point)
          next_time=np.array(next_time).reshape(5,1)  
          final.index = list(next_time.flatten())
          final.index.name = "DateTime"
          st.dataframe(final)
          st.subheader('Model Error Rate')
          st.write("Mean Absolute Error (MAE) =",MAE)
          st.write("Mean Absolute Percentage Error (MAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MAPE), unsafe_allow_html=True)
          st.write("Median Absolute Percentage Error (MDAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MDAPE), unsafe_allow_html=True)


 #12
    if time_interval2 == "15 minutes" and stocks =="BAJAJ-AUTO.NS" and price == "Adj Close" and models == "GRU":
          M=load_model("MODELS/GRU_bajaj_adj(15m).h5")
          DF=yf.download(stocks,period='60d',interval='15m')
          fig ,valid,MAE,MAPE,MDAPE ,final= plot_model(M, DF, stocks, price)
          st.subheader(f'Predictions from {models} Model')
          if __name__ == "__main__":
              last_time_point = DF.index[-1].to_pydatetime()  # Replace with your specific last time index
              next_time = get_time(last_time_point)
          next_time=np.array(next_time).reshape(5,1)  
          final.index = list(next_time.flatten())
          final.index.name = "DateTime"
          st.dataframe(final)
          st.subheader('Model Error Rate')
          st.write("Mean Absolute Error (MAE) =",MAE)
          st.write("Mean Absolute Percentage Error (MAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MAPE), unsafe_allow_html=True)
          st.write("Median Absolute Percentage Error (MDAPE) = <span style='color:green'>{:.2f}&nbsp;%</span>".format(MDAPE), unsafe_allow_html=True)
