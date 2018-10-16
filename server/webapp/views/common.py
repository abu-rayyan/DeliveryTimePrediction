from flask import Response, request
from webapp import app
from flask_cors import CORS
from flask_autodoc import Autodoc
import pandas as pd 
import json
import numpy as np
import math
from sklearn.externals import joblib
import datetime
CORS(app)
#auto = Autodoc(app)


@app.route('/')
def index_route():
    return '/'


@app.route('/predict-time', methods=['GET', 'POST'])
#@auto.doc()
def return_predicted_time():

    # Recieve input in Json format which is coming from POSTMAN request
    json_array = request.get_json()
    #Convert json into pandas dataframe
    df = pd.DataFrame.from_dict(json_array)
    input_items = len(json_array)
    if input_items == 4:
        model, inputs, time = model1(json_array)
    elif input_items == 6:
        model, inputs, time = model2(json_array)
    elif input_items == 7:
        model, inputs, time = model3(json_array)
    elif input_items == 8:
        model, inputs, time = model4(json_array)
    else:
        print("Please lookafter your inputs")
    
    # predict the delivery time
    predict_output = model.predict(inputs)
    # predicted time is in epoc date so we will convert again to 
    minutes = np.floor(np.multiply(np.subtract(np.multiply(predict_output,24),
                                        np.floor(np.multiply(predict_output,24))),60))
    
    hours = str(predict_output[0]).split('.')[0]
    minutes = str(minutes[0]).split('.')[0]
    
    total_time = time + datetime.timedelta(hours=int(hours) , minutes=int(minutes))
    total_time_into_datetime = total_time.to_pydatetime()
    
    if int(total_time_into_datetime.hour) >= 21:
        hours = int(hours)+10
        total_time = time.replace(hour=hours, minute=int(minutes))
        total_time = total_time + datetime.timedelta(days=1)
        if int(total_time_into_datetime.weekday()) == 4:
            total_time = total_time + datetime.timedelta(days=2)
        
    hour_min_days = total_time - time
    t1={ 
        'time_in_hours_and_minutes':str(hour_min_days),
        'time_with_date_and_time':str(total_time)
    
    }

    response = Response(json.dumps(t1))
    return response

def model1(json_array):
    df = pd.DataFrame.from_dict(json_array)
    df['CreatedTime'] = pd.to_datetime(df['CreatedTime'])
    timestamp_into_int = df['CreatedTime'].values.astype(int)
    location_code = df[['LocationCode', 'PendingOrdersLocationWise', 'Qty']].values
    X = np.insert(location_code, 1,timestamp_into_int , axis=1)
    model = joblib.load('/home/expertsvision/Desktop/delivery_predict_time/model1_dt.sav')
    time = df.iloc[0]['CreatedTime']
    return model, X, time

def model2(json_array):
    df = pd.DataFrame.from_dict(json_array)
    df[['CreatedTime','BikerAssignedTime']] = df[['CreatedTime',
                                                'BikerAssignedTime']].apply(pd.to_datetime)
    timestamp_into_int = df[['CreatedTime','BikerAssignedTime']].values.astype(int)
    df['biker_id'], levels = pd.factorize(df['Biker'])
    other_features = df['BikerID', 'PendingOrderByBiker', 'PendingOrdersLocationWise', 'Qty'].values                                   
    X = np.concatenate((other_features, timestamp_into_int), axis=1)
    model = joblib.load('/home/expertsvision/Desktop/delivery_predict_time/model2_dt.sav')
    time =  df.iloc[0]['BikerAssignedTime']
    return model, X, time

def model3(json_array):
    df = pd.DataFrame.from_dict(json_array)
    df[['CreatedTime','BikerAssignedTime', 'BikerAcceptedTime']] = df[['CreatedTime',
                        'BikerAssignedTime', 'BikerAcceptedTime']].apply(pd.to_datetime)
    timestamp_into_int = df[['CreatedTime','BikerAssignedTime', 'BikerAcceptedTime']].values.astype(int)
    other_features = df['BikerID', 'PendingOrderByBiker', 'PendingOrdersLocationWise', 'Qty'].values 
    X = np.concatenate((other_features,timestamp_into_int), axis=1)                       
    model = joblib.load('/home/expertsvision/Desktop/delivery_predict_time/model3_dt.sav')
    time =  df.iloc[0]['BikerAcceptedTime']
    return model, X, time

def model4(json_array):
    df = pd.DataFrame.from_dict(json_array)
    df[['CreatedTime','BikerAssignedTime', 'BikerAcceptedTime', 'InBikeTime']] = df[['CreatedTime',
            'BikerAssignedTime', 'BikerAcceptedTime', 'InBikeTime']].apply(pd.to_datetime)
    timestamp_into_int = df[['CreatedTime','BikerAssignedTime', 'BikerAcceptedTime', 
                                                            'InBikeTime']].values.astype(int)
    
    other_features = df['BikerID', 'PendingOrderByBiker', 'PendingOrdersLocationWise', 'Qty'].values
    X = np.concatenate((other_features, timestamp_into_int), axis=1)
    model = joblib.load('/home/expertsvision/Desktop/delivery_predict_time/model4_dt.sav')
    time =  df.iloc[0]['InBikeTime']
    return model, X, time

def get_time(x, c1, c2):     #find the difference between two date columns of dataframe x
    diff =  x[c2] - x[c1]
    days = diff.days
    days_to_hours = days * 24
    diff_btw_two_times = (diff.seconds) / 3600
    overall_hours = days_to_hours + diff_btw_two_times
    return overall_hours

@app.route('/docs')
def return_api_docs():
    """
    api docs route
    :return:
    """
    return auto.html()