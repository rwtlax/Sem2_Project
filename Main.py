import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
import streamlit as st
from mftool import Mftool
import time
from pmdarima.model_selection import train_test_split
from pmdarima import metrics

import streamlit_authenticator as stauth
import sqlite3

st.set_page_config(
    page_title="NAV Prediction App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

mf = Mftool()
data1 = mf.get_scheme_codes()
codes = pd.DataFrame(data1.keys())
codes['MF'] = pd.DataFrame(data1.values())


tab1, tab2, tab3 = st.tabs(["Home","Prediction","Trend Analysis"])

with tab1:
    st.header("Mutual Fund Prediction")
    st.subheader ("This tool will aid in understanding the possible further movements of nav and the past trends.")
    st.text("You can sign up on the Trend Analysis tab to get the MF with highest returns. Once you login you will also be able to see the 50-day moving average trend and 200 day moving average trend.")
    st.text("Crossovers of 50 DMA and 200 DMA have great significances.")
    st.text("There are two terms named according to these crossovesr")
    st.text("Golden Cross: When 50 DMA is crossing above 200 DMA it is known as GOLDEN CROSS. When this happens it is best to buy because after that the nav will start for long term uptrend")
    st.text("Death Cross: When 50 DMA is crossing below 200 DMA it is known as DEATH CROSS. When this happens it is best time to exit because after that the nav will start for long term downtrend")
with tab2:

    st.title("Mutual Fund's NAV Prediction")
    MF = st.selectbox("Select a Mutual Fund", data1.values())


    def get_key_from_values(data, val):
        keys = [k for k, v in data.items() if v == val]
        if keys:
            return keys[0]


    key = get_key_from_values(data1, MF)
    data = mf.get_scheme_historical_nav(key)
    navs = pd.DataFrame(data['data'])
    navs['date'] = pd.to_datetime(navs['date'], format='%d-%m-%Y')
    navs['nav'] = navs['nav'].astype(float)
    navs = navs.loc[::-1].reset_index(drop=True)

    x_train, x_test= train_test_split(navs['nav'], test_size=0.2)

    data_copy = pd.DataFrame(data['data'])
    data_copy['date'] = pd.to_datetime(data_copy['date'], format='%d-%m-%Y')
    data_copy['nav'] = data_copy['nav'].astype(float)
    data_copy = data_copy[::-1].reset_index(drop=True)

    detail = mf.get_scheme_details(key)
    d = pd.DataFrame(detail)
    d['start_nav'] = detail['scheme_start_date']['nav']
    d.drop('nav', axis=0, inplace=True)
    d.to_string(index=False)
    st.header("Scheme Details")
    st.table(d)

    with st.spinner('Wait for it...'):
        time.sleep(10)
    st.success('Done!')

    model = pm.auto_arima(x_train)

# for testing data
    n_periods_test = len(x_test)
    fc_test = model.predict(n_periods=n_periods_test)

    navs['Pred'] = fc_test

    model1 = pm.auto_arima(navs['nav'])
    n_periods = 20
    fc, confint = model1.predict(n_periods=n_periods, return_conf_int=True)

# make series for plotting purpose
    index_of_fc = np.arange(len(navs.values), len(navs.values) + n_periods)
    fc_series = pd.Series(fc, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    time = pd.date_range(navs['date'].iloc[-1], periods=20, inclusive='right')
    dates = pd.DataFrame()
    dates['date'] = pd.DataFrame(time)
    dates['nav']=pd.DataFrame(fc_series.values.reshape(-1,1))
    dates['date'] = pd.to_datetime(dates['date'], format='%d-%m-%Y')
    dates['lower_limit'] = pd.DataFrame(lower_series.values.reshape(-1,1))
    dates['upper_limit'] = pd.DataFrame(upper_series.values.reshape(-1,1))


    st.header("Accuracy")
    value = metrics.smape(x_test,fc_test)
    st.subheader(value)

    st.text("The model's forcast is acceptable if the accuracy score is below 25")

# Plot
    col1, col2 = st.columns(2)
    with col1:
        fig = plt.figure(figsize=(12, 6))
        plt.plot(navs['date'], navs['nav'])
        plt.plot(navs['date'],navs['Pred'],color='red')
        plt.plot(dates['date'], dates['nav'], color='darkgreen')
        plt.title("Mutual Fund's nav over time")
        plt.ylabel("Price")
        plt.xlabel("Date")
        plt.legend(["Actual","Test Data","Predicted Values"],loc="upper left")
        plt.fill_between(dates['date'],
                 dates['lower_limit'],
                 dates['upper_limit'],
                 color='k', alpha=.15)
        st.pyplot(fig)
    with col2:
        fig1 = plt.figure(figsize=(12, 6))
        plt.plot(dates['date'], dates['nav'], color='darkgreen')
        plt.ylabel("Price")
        plt.xlabel("Date")
        plt.title('Predicted Values Along With Confidence Interval')
        plt.fill_between(dates['date'],
                 dates['lower_limit'],
                 dates['upper_limit'],
                 color='k', alpha=.15)
        st.pyplot(fig1)


    st.header("Predicted Values")
    st.table(dates)

with tab3:
    # Connect to the database
    conn = sqlite3.connect('Users.db')

    # read the data from the "UserTable" table into a DataFrame object
    cur = conn.cursor()
    df = pd.read_sql_query("SELECT * from UserTable", conn)


    names = df['Name'].tolist()
    usernames = df['Username'].tolist()
    passwords = df['Password'].tolist()


    # Close the database connection
    conn.close()

    hashed_passwords = stauth.Hasher(passwords).generate()

    authenticator = stauth.Authenticate(names,usernames,hashed_passwords,'some_cookie','random_key', cookie_expiry_days=1)
    name,authentication_status,username = authenticator.login('Login', 'main')

    if st.session_state['authentication_status']:
        col1, col2, col3 = st.columns([5,7,1])
        with col3:
            authenticator.logout('Logout', 'main')
        with col1:
            st.title('Welcome *%s*' % (st.session_state['name']))

        age = int(st.number_input('Age'))
        proportion = 100 - age
        if 20 <= age < 30:
            st.metric(label="Invest following proportion in High Risk MF", value=proportion)
        elif 30 <= age < 40:
            st.metric(label="Invest following proportion in Moderately High Risk MF", value=proportion)
        elif 40 <= age < 50:
            st.metric(label="Invest following proportion in Balanced MF", value=proportion)
        elif 50 <= age < 60:
            st.metric(label="Invest following proportion in Moderately Low Risk MF", value=proportion)
        elif age >= 60:
            st.metric(label="Invest following proportion in Low Risk MF", value=proportion)
        else:
            st.metric(label="Invest following proportion in High Risk MF", value=proportion)


        selected_values = st.multiselect("Select a few Mutual Funds to choose from: ", data1.values(), default=None)
        average = []
        value = []
        for i in range(len(selected_values)):

            def get_key_from_values(data, val):
                keys = [k for k, v in data.items() if v == val]
                if keys:
                    return keys[0]


            key = get_key_from_values(data1, selected_values[i])
            data = mf.get_scheme_historical_nav(key)
            navs = pd.DataFrame(data['data'])
            navs['date'] = pd.to_datetime(navs['date'], format='%d-%m-%Y')
            navs['nav'] = navs['nav'].astype(float)
            navs = navs.loc[::-1].reset_index(drop=True)

            x_train, x_test= train_test_split(navs['nav'], test_size=0.2)

            data_copy = pd.DataFrame(data['data'])
            data_copy['date'] = pd.to_datetime(data_copy['date'], format='%d-%m-%Y')
            data_copy['nav'] = data_copy['nav'].astype(float)
            data_copy = data_copy[::-1].reset_index(drop=True)

            model = pm.auto_arima(x_train)

            # for testing data
            n_periods_test = len(x_test)
            fc_test = model.predict(n_periods=n_periods_test)

            navs['Pred'] = fc_test

            model1 = pm.auto_arima(navs['nav'])
            n_periods = 20
            fc, confint = model1.predict(n_periods=n_periods, return_conf_int=True)

            # make series for plotting purpose
            index_of_fc = np.arange(len(navs.values), len(navs.values) + n_periods)
            fc_series = pd.Series(fc, index=index_of_fc)
            lower_series = pd.Series(confint[:, 0], index=index_of_fc)
            upper_series = pd.Series(confint[:, 1], index=index_of_fc)

            time = pd.date_range(navs['date'].iloc[-1], periods=20, inclusive='right')
            dates_i = pd.DataFrame()
            dates_i['date'] = pd.DataFrame(time)
            dates_i['nav'] = pd.DataFrame(fc_series.values.reshape(-1, 1))
            dates_i['date'] = pd.to_datetime(dates['date'], format='%d-%m-%Y')
            dates_i['lower_limit'] = pd.DataFrame(lower_series.values.reshape(-1, 1))
            dates_i['upper_limit'] = pd.DataFrame(upper_series.values.reshape(-1, 1))
            avg = int(dates_i['nav'].mean())
            returns = float(detail['scheme_start_date']['nav'])
            return_value = (avg/returns) - 1
            average.append(return_value)
            value.append(metrics.smape(x_test, fc_test))


        if len(average) != 0:
            index1 = average.index(max(average))
            index2 = value.index(min(value))


            st.header("MF with highest return is : ")
            st.header(selected_values[index1])
            key = get_key_from_values(data1, selected_values[index1])
            data = mf.get_scheme_historical_nav(key)
            navs = pd.DataFrame(data['data'])
            navs['date'] = pd.to_datetime(navs['date'], format='%d-%m-%Y')
            navs['nav'] = navs['nav'].astype(float)
            navs = navs.loc[::-1].reset_index(drop=True)

            x_train, x_test = train_test_split(navs['nav'], test_size=0.2)

            data_copy = pd.DataFrame(data['data'])
            data_copy['date'] = pd.to_datetime(data_copy['date'], format='%d-%m-%Y')
            data_copy['nav'] = data_copy['nav'].astype(float)
            data_copy = data_copy[::-1].reset_index(drop=True)

            detail = mf.get_scheme_details(key)
            d = pd.DataFrame(detail)
            d['start_nav'] = detail['scheme_start_date']['nav']
            d.drop('nav', axis=0, inplace=True)
            d.to_string(index=False)
            st.header("Scheme Details")
            st.table(d)

            model = pm.auto_arima(x_train)

            # for testing data
            n_periods_test = len(x_test)
            fc_test = model.predict(n_periods=n_periods_test)

            navs['Pred'] = fc_test

            model1 = pm.auto_arima(navs['nav'])
            n_periods = 20
            fc, confint = model1.predict(n_periods=n_periods, return_conf_int=True)

            # make series for plotting purpose
            index_of_fc = np.arange(len(navs.values), len(navs.values) + n_periods)
            fc_series = pd.Series(fc, index=index_of_fc)
            lower_series = pd.Series(confint[:, 0], index=index_of_fc)
            upper_series = pd.Series(confint[:, 1], index=index_of_fc)

            time = pd.date_range(navs['date'].iloc[-1], periods=20, inclusive='right')
            dates = pd.DataFrame()
            dates['date'] = pd.DataFrame(time)
            dates['nav'] = pd.DataFrame(fc_series.values.reshape(-1, 1))
            dates['date'] = pd.to_datetime(dates['date'], format='%d-%m-%Y')
            dates['lower_limit'] = pd.DataFrame(lower_series.values.reshape(-1, 1))
            dates['upper_limit'] = pd.DataFrame(upper_series.values.reshape(-1, 1))

            st.header("Accuracy")
            value = metrics.smape(x_test, fc_test)
            st.subheader(value)

            st.text("The model's forcast is acceptable if the accuracy score is below 25")
            navs['DMA50'] = navs['nav'].rolling(50).mean()
            navs['DMA200'] = navs['nav'].rolling(200).mean()
            fig2 = plt.figure(figsize=(12, 6))
            plt.plot(navs['date'],navs['nav'])
            plt.plot(navs['date'], navs['DMA50'], color='red')
            plt.plot(navs['date'], navs['DMA200'], color='darkgreen')
            plt.legend(["Price", "50 DMA", "200 DMA"], loc="upper left")
            st.pyplot(fig2)
            # Plot
            col1, col2 = st.columns(2)
            with col1:
                fig = plt.figure(figsize=(12, 6))
                plt.plot(navs['date'], navs['nav'])
                plt.plot(navs['date'], navs['Pred'], color='red')
                plt.plot(dates['date'], dates['nav'], color='darkgreen')
                plt.title("Mutual Fund's nav over time")
                plt.ylabel("Price")
                plt.xlabel("Date")
                plt.legend(["Actual", "Test Data", "Predicted Values"], loc="upper left")
                plt.fill_between(dates['date'],
                             dates['lower_limit'],
                             dates['upper_limit'],
                             color='k', alpha=.15)
                st.pyplot(fig)
            with col2:
                fig1 = plt.figure(figsize=(12, 6))
                plt.plot(dates['date'], dates['nav'], color='darkgreen')
                plt.ylabel("Price")
                plt.xlabel("Date")
                plt.title('Predicted Values Along With Confidence Interval')
                plt.fill_between(dates['date'],
                             dates['lower_limit'],
                             dates['upper_limit'],
                             color='k', alpha=.15)
                st.pyplot(fig1)

            st.header("MF with highest accuracy is: ")
            st.header(selected_values[index2])
            key = get_key_from_values(data1, selected_values[index2])
            data = mf.get_scheme_historical_nav(key)
            navs = pd.DataFrame(data['data'])
            navs['date'] = pd.to_datetime(navs['date'], format='%d-%m-%Y')
            navs['nav'] = navs['nav'].astype(float)
            navs = navs.loc[::-1].reset_index(drop=True)

            x_train, x_test = train_test_split(navs['nav'], test_size=0.2)

            data_copy = pd.DataFrame(data['data'])
            data_copy['date'] = pd.to_datetime(data_copy['date'], format='%d-%m-%Y')
            data_copy['nav'] = data_copy['nav'].astype(float)
            data_copy = data_copy[::-1].reset_index(drop=True)

            detail = mf.get_scheme_details(key)
            d = pd.DataFrame(detail)
            d['start_nav'] = detail['scheme_start_date']['nav']
            d.drop('nav', axis=0, inplace=True)
            d.to_string(index=False)
            st.header("Scheme Details")
            st.table(d)

            model = pm.auto_arima(x_train)

            # for testing data
            n_periods_test = len(x_test)
            fc_test = model.predict(n_periods=n_periods_test)

            navs['Pred'] = fc_test

            model1 = pm.auto_arima(navs['nav'])
            n_periods = 20
            fc, confint = model1.predict(n_periods=n_periods, return_conf_int=True)

            # make series for plotting purpose
            index_of_fc = np.arange(len(navs.values), len(navs.values) + n_periods)
            fc_series = pd.Series(fc, index=index_of_fc)
            lower_series = pd.Series(confint[:, 0], index=index_of_fc)
            upper_series = pd.Series(confint[:, 1], index=index_of_fc)

            time = pd.date_range(navs['date'].iloc[-1], periods=20, inclusive='right')
            dates = pd.DataFrame()
            dates['date'] = pd.DataFrame(time)
            dates['nav'] = pd.DataFrame(fc_series.values.reshape(-1, 1))
            dates['date'] = pd.to_datetime(dates['date'], format='%d-%m-%Y')
            dates['lower_limit'] = pd.DataFrame(lower_series.values.reshape(-1, 1))
            dates['upper_limit'] = pd.DataFrame(upper_series.values.reshape(-1, 1))

            st.header("Accuracy")
            value = metrics.smape(x_test, fc_test)
            st.subheader(value)

            st.text("The model's forcast is acceptable if the accuracy score is below 25")
            navs['DMA50'] = navs['nav'].rolling(50).mean()
            navs['DMA200'] = navs['nav'].rolling(200).mean()
            fig3 = plt.figure(figsize=(12, 6))
            plt.plot(navs['date'],navs['nav'])
            plt.plot(navs['date'], navs['DMA50'], color='red')
            plt.plot(navs['date'], navs['DMA200'], color='darkgreen')
            st.pyplot(fig3)

        # Plot
            col1, col2 = st.columns(2)
            with col1:
                fig = plt.figure(figsize=(12, 6))
                plt.plot(navs['date'], navs['nav'])
                plt.plot(navs['date'], navs['Pred'], color='red')
                plt.plot(dates['date'], dates['nav'], color='darkgreen')
                plt.title("Mutual Fund's nav over time")
                plt.ylabel("Price")
                plt.xlabel("Date")
                plt.legend(["Actual", "Test Data", "Predicted Values"], loc="upper left")
                plt.fill_between(dates['date'],
                             dates['lower_limit'],
                             dates['upper_limit'],
                             color='k', alpha=.15)
                st.pyplot(fig)
            with col2:
                fig1 = plt.figure(figsize=(12, 6))
                plt.plot(dates['date'], dates['nav'], color='darkgreen')
                plt.ylabel("Price")
                plt.xlabel("Date")
                plt.title('Predicted Values Along With Confidence Interval')
                plt.fill_between(dates['date'],
                             dates['lower_limit'],
                             dates['upper_limit'],
                             color='k', alpha=.15)
                st.pyplot(fig1)

    elif st.session_state['authentication_status'] == False:
        st.error('Username/password is incorrect')
    elif st.session_state['authentication_status'] == None:
        st.warning('Please enter your username and password')
    
    val = st.button("Sign Up")

    if val:
        name = st.text_input ("Enter your name: ")
        usr = st.text_input ("Enter Username: ")
        pwd = st.text_input ("Enter Password: ")

        signup = st.button("Submit")

        if signup:
                
            conn = sqlite3.connect('Users.db')

            # read the data from the "UserTable" table into a DataFrame object
            cur = conn.cursor()

            insert_record = """INSERT INTO UserTable(Name, Username, Password) VALUES (name,usr,pwd)"""

            cur.execute(insert_record)
            conn.close()
        



