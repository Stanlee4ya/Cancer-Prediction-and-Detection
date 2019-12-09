from flask import Flask, abort, jsonify, request, render_template, redirect, url_for, session
from sklearn.externals import joblib
import pickle
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import json
import math
import numpy as np
import pandas as pd
from sklearn import svm
import os
from flask import send_from_directory
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
from keras import optimizers
from bokeh.embed import components
from bokeh.layouts import column, gridplot, layout, row
from bokeh.models import ColumnDataSource, HoverTool, PrintfTickFormatter
from bokeh.plotting import figure
from bokeh.transform import factor_cmap

df = pd.read_csv('../data/cervical_survival.csv')
df1=pd.read_csv('../data/prostate_survival.csv')

model = joblib.load('cervical_surv_model.pkl')
model1 = joblib.load('prostate_surv_model.pkl')
model2 = pickle.load(open('cervival_risk_model.pkl', 'rb'))
model3 = pickle.load(open('prostate_risk_model.pkl', 'rb'))
dir_path = os.path.dirname(os.path.realpath(__file__))
# UPLOAD_FOLDER = dir_path + '/uploads'
# STATIC_FOLDER = dir_path + '/static'
UPLOAD_FOLDER = 'uploads'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

graph= tf.compat.v1.get_default_graph()
#graph = tf.get_default_graph()
with graph.as_default():
    # load model at very first
    model5 = load_model(STATIC_FOLDER + '/' + 'prostate.h5')
    model6 = load_model(STATIC_FOLDER + '/' + 'cervical.h5')
    model5.compile(optimizer = optimizers.RMSprop(lr=1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model6.compile(optimizer = optimizers.RMSprop(lr=1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

################################################################################
#                             - END OF IMPORTS -                               #
################################################################################


################################################################################
#                               > CONSTANT VALUES                              #
################################################################################
palette = ['#17202A', '#566573', '#21618C', '#D2B4DE']

chart_font = 'Helvetica'
chart_title_font_size = '16pt'
chart_title_alignment = 'center'
axis_label_size = '14pt'
axis_ticks_size = '12pt'
default_padding = 30
chart_inner_left_padding = 0.015
chart_font_style_title = 'bold italic'

################################################################################
#                          - END OF CONSTANT VALUES -                          #
################################################################################


################################################################################
#                             > HELPER FUNCTIONS                               #
################################################################################
def palette_generator(length, palette):
    int_div = length // len(palette)
    remainder = length % len(palette)
    return (palette * int_div) + palette[:remainder]


def plot_styler(p):
    p.title.text_font_size = chart_title_font_size
    p.title.text_font  = chart_font
    p.title.align = chart_title_alignment
    p.title.text_font_style = chart_font_style_title
    p.y_range.start = 0
    p.x_range.range_padding = chart_inner_left_padding
    p.xaxis.axis_label_text_font = chart_font
    p.xaxis.major_label_text_font = chart_font
    p.xaxis.axis_label_standoff = default_padding
    p.xaxis.axis_label_text_font_size = axis_label_size
    p.xaxis.major_label_text_font_size = axis_ticks_size
    p.yaxis.axis_label_text_font = chart_font
    p.yaxis.major_label_text_font = chart_font
    p.yaxis.axis_label_text_font_size = axis_label_size
    p.yaxis.major_label_text_font_size = axis_ticks_size
    p.yaxis.axis_label_standoff = default_padding
    p.toolbar.logo = None
    p.toolbar_location = None


def redraw(stage):
    survived_chart = survived_bar_chart(df, stage)
    title_chart = class_titles_bar_chart(df, stage)
    hist_age = age_hist(df, stage)
    return (
        survived_chart,
        title_chart,
        hist_age
    )

################################################################################
#                          - END OF HELPER FUNCTIONS -                         #
################################################################################
################################################################################
#                               > CONSTANT VALUES                              #
################################################################################
palette = ['#17202A', '#566573', '#21618C', '#D2B4DE']

chart_font = 'Helvetica'
chart_title_font_size = '16pt'
chart_title_alignment = 'center'
axis_label_size = '14pt'
axis_ticks_size = '12pt'
default_padding = 30
chart_inner_left_padding = 0.015
chart_font_style_title = 'bold italic'

################################################################################
#                          - END OF CONSTANT VALUES -                          #
################################################################################


################################################################################
#                             > HELPER FUNCTIONS                               #
################################################################################
def palette_generator(length, palette):
    int_div = length // len(palette)
    remainder = length % len(palette)
    return (palette * int_div) + palette[:remainder]


def plot_styler(p):
    p.title.text_font_size = chart_title_font_size
    p.title.text_font  = chart_font
    p.title.align = chart_title_alignment
    p.title.text_font_style = chart_font_style_title
    p.y_range.start = 0
    p.x_range.range_padding = chart_inner_left_padding
    p.xaxis.axis_label_text_font = chart_font
    p.xaxis.major_label_text_font = chart_font
    p.xaxis.axis_label_standoff = default_padding
    p.xaxis.axis_label_text_font_size = axis_label_size
    p.xaxis.major_label_text_font_size = axis_ticks_size
    p.yaxis.axis_label_text_font = chart_font
    p.yaxis.major_label_text_font = chart_font
    p.yaxis.axis_label_text_font_size = axis_label_size
    p.yaxis.major_label_text_font_size = axis_ticks_size
    p.yaxis.axis_label_standoff = default_padding
    p.toolbar.logo = None
    p.toolbar_location = None


def redraw(stage):
    survived_chart = survived_bar_chart(df1, stage)
    title_chart = class_titles_bar_chart(df1, stage)
    hist_age = age_hist(df1, stage)
    return (
        survived_chart,
        title_chart,
        hist_age
    )

################################################################################
#                               > MAIN ROUTE                                   #
################################################################################

app = Flask(__name__)

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = 'cancermed'

# Enter your database connection details below
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'cancerlogin'

# Intialize MySQL
mysql = MySQL(app)

# http://localhost:5000/pythonlogin/ - this will be the login page, we need to use both GET and POST requests
@app.route('/', methods=['GET', 'POST'])
def login():
   # Output message if something goes wrong...
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
         # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password))
        # Fetch one record and return result
        account = cursor.fetchone()
         # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            # Redirect to home page
            return redirect(url_for('home'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'



    return render_template('index.html', msg='')

# http://localhost:5000/python/logout - this will be the logout page
@app.route('/logout')
# http://localhost:5000/pythinlogin/register - this will be the registration page, we need to use both GET and POST requests
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('login'))
# Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response
@app.route('/register', methods=['GET', 'POST'])
def register():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
         # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        
        # If account exists show error and validation checks
        
        if not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s, %s)', (username, password, email))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
    # Show registration form with message (if any)
    return render_template('register.html', msg=msg)
# http://localhost:5000/pythinlogin/home - this will be the home page, only accessible for loggedin users
@app.route('/home')
def home():
    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the home page
        return render_template('index3.html')
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))
@app.route('/prostate_surv')
def prostate_surv():
    return render_template('prostate_surv_index.html')


# @app.route('/api', methods=['POST'])
# def make_prediction():
#     data = request.get_json(force=True)
#     #convert our json to a numpy array
#     one_hot_data = input_to_one_hot(data)
#     predict_request = gbr.predict([one_hot_data])
#     output = [predict_request[0]]
#     print(data)
#     return jsonify(results=output)

def input_to_one_hot1(data):
    # initialize the target vector with zero values
    enc_input = np.zeros(12)
    # set the numerical input as they are
    enc_input[0] = data['gleason_score']
    enc_input[1] = data['n_score']
    enc_input[2] = data['race']
    enc_input[3] = data['weight']
    enc_input[4] = data['tumor_1_year']
    enc_input[5] = data['psa_1_year']
    enc_input[6] = data['rd_thrpy']
    enc_input[7] = data['h_thrpy']
    enc_input[8] = data['chm_thrpy']
    enc_input[9] = data['brch_thrpy']
    enc_input[10] = data['rad_rem']
    enc_input[11] = data['multi_thrpy']
   
    return enc_input

@app.route('/prostate_survapi',methods=['POST'])
def get_delay():
    result=request.form
    gleason_score = result['gleason_score']
    n_score = result['n_score']
    race= result['race']
    weight = result['weight']
    tumor_1_year=result['tumor_1_year']
    psa_1_year = result['psa_1_year']
    rd_thrpy = result['rd_thrpy']
    h_thrpy = result['h_thrpy']
    chm_thrpy = result['chm_thrpy']
    brch_thrpy = result['brch_thrpy']
    rad_rem = result['rad_rem']
    multi_thrpy = result['multi_thrpy']

    user_input ={'gleason_score':gleason_score, 'n_score':n_score, 'race':race, 'weight':weight, 'tumor_1_year':tumor_1_year, 'psa_1_year':psa_1_year,'rd_thrpy':rd_thrpy, 'h_thrpy':h_thrpy, 'chm_thrpy':chm_thrpy, 'brch_thrpy':brch_thrpy, 'rad_rem':rad_rem, 'multi_thrpy':multi_thrpy}
    
    print(user_input)
    a = input_to_one_hot1(user_input)
    month_pred = model1.predict([a])[0]
    month_pred = round(month_pred, 2)
    return json.dumps({'time_of_survival':month_pred});
    # return render_template('result.html',prediction=price_pred)
#----------------------end of prostate survival prediction-------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#cervical survival predictions
@app.route('/surv_cervical')
def surv_cervical():
    return render_template('cervical_surv_index.html')


# @app.route('/api', methods=['POST'])
# def make_prediction():
#     data = request.get_json(force=True)
#     #convert our json to a numpy array
#     one_hot_data = input_to_one_hot(data)
#     predict_request = gbr.predict([one_hot_data])
#     output = [predict_request[0]]
#     print(data)
#     return jsonify(results=output)

def input_to_one_hot(new_data):
    # initialize the target vector with zero values
    enc_input = np.zeros(15)
    # set the numerical input as they are
    enc_input[0] = new_data['TIME_RTX']
    enc_input[1] = new_data['TIME_BTX']
    enc_input[2] = new_data['META_CODE']
    enc_input[3] = new_data['SURGERY']
    enc_input[4] = new_data['STAGING_CODE']
    enc_input[5] = new_data['READ_CODE']
    enc_input[6] = new_data['THERAPY_CODE']
    enc_input[7] = new_data['OCUP_CODE']
    enc_input[8] = new_data['AGE_CODIFY']
    enc_input[9] = new_data['HISTOPATHOLOGICAL_REPORT_CODE']
    enc_input[10] = new_data['COMORBITIES_CODE']
    enc_input[11] = new_data['NUMBER_OF_CHILDREN_CODE']
    enc_input[12] = new_data['SMOKING_CODE']
    enc_input[13] = new_data[ 'MARITAL_STATUS']
    enc_input[14] = new_data[ 'OUTCOME_COD']
   
   
    return enc_input


@app.route('/surv_cervicalapi',methods=['POST'])
def cervcal():
    result=request.form
    TIME_RTX= result['TIME_RTX']
    TIME_BTX= result['TIME_BTX']
    META_CODE= result['META_CODE']
    SURGERY=result['SURGERY']
    STAGING_CODE= result['STAGING_CODE']
    READ_CODE= result['READ_CODE']
    THERAPY_CODE= result['THERAPY_CODE']
    OCUP_CODE= result['OCUP_CODE']
    AGE_CODIFY= result['AGE_CODIFY']
    HISTOPATHOLOGICAL_REPORT_CODE= result['HISTOPATHOLOGICAL_REPORT_CODE']
    COMORBITIES_CODE= result['COMORBITIES_CODE']
    NUMBER_OF_CHILDREN_CODE= result['NUMBER_OF_CHILDREN_CODE']
    SMOKING_CODE= result['SMOKING_CODE']
    MARITAL_STATUS= result['MARITAL_STATUS']
    OUTCOME_COD= result['OUTCOME_COD']
  


    user_input ={'TIME_RTX':TIME_RTX, 'TIME_BTX':TIME_BTX, 'META_CODE':META_CODE, 'SURGERY':SURGERY, 
             'STAGING_CODE':STAGING_CODE,'READ_CODE':READ_CODE, 'THERAPY_CODE':THERAPY_CODE,
             
             'OCUP_CODE':OCUP_CODE, 'AGE_CODIFY':AGE_CODIFY, 'HISTOPATHOLOGICAL_REPORT_CODE':HISTOPATHOLOGICAL_REPORT_CODE,
             'COMORBITIES_CODE':COMORBITIES_CODE, 'NUMBER_OF_CHILDREN_CODE':NUMBER_OF_CHILDREN_CODE, 'SMOKING_CODE':SMOKING_CODE, 'MARITAL_STATUS':MARITAL_STATUS
             , 'OUTCOME_COD':OUTCOME_COD}
    
    print(user_input)
    a = input_to_one_hot(user_input)
    month_pred = model.predict([a])[0]
    listtype = month_pred.tolist() 
    
    return json.dumps({'TIME_OF_SURVIVAL':listtype});
    # return render_template('result.html',prediction=price_pred)
#----------------------end of cervical survival prediction-------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#cervival risk predictions
@app.route('/cervical_risk')
def cervical_risk():
    return render_template('cervical_risk_index.html')

@app.route('/cervical_risk_predict',methods=['POST'])
def cervical_risk_predict():
    '''
    For rendering results on HTML GUI
    '''


    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model2.predict_proba(final_features)[:,1]

    output =np.round(prediction[0], 1)
    prob=output*100

    return render_template('cervical_risk_predict.html', prediction_text='Risk of developing Cervical Cancer is {}'.format(prob))

@app.route('/cervical_risk_predict_api',methods=['POST'])
def cervical_risk_predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model2.predict_proba([np.array(list(data.values()))])[:,1]

    output = prediction[0]
    return jsonify(output)
#--------------end of cervical risk predictions--------------------------------------
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
#prostate risk predictions

@app.route('/risk_prostate')
def risk_prostate():
    return render_template('prostate_risk_index.html')

@app.route('/prostate_risk_predict',methods=['POST'])
def prostate_risk_predict():
    '''
    For rendering results on HTML GUI
    '''
    #model = pickle.load(open('mode.pkl','rb'))
    #prediction = model.predict_proba([[0.737164,3.473518,64,0.615186,0,-1.386294,6,0,0.765468]])[:,1]
    #print(prediction)
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model3.predict_proba(final_features)[:,1]


    

    output = round(prediction[0], 2)
    perc=output*100

    return render_template('prostate_risk_predict.html', prediction_text='Prostate Cancer Risk is {}'.format(perc) + '%')

@app.route('/prostate_risk_predict_api',methods=['POST'])
def prostate_risk_predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model3.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
#-----------------end of prostate risk predictions------------------------------------------
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#cervical survival visualisations


@app.route('/chart', methods=['GET', 'POST'])
def chart():
    selected_class = request.form.get('dropdown-select')

    if selected_class == 0 or selected_class == None:
        survived_chart, title_chart, hist_age = redraw(1)
    else:
        survived_chart, title_chart, hist_age = redraw(selected_class)

    script_survived_chart, div_survived_chart = components(survived_chart)
    script_title_chart, div_title_chart = components(title_chart)
    script_hist_age, div_hist_age = components(hist_age)

    return render_template(
        'cervical_visu_index.html',
        div_survived_chart=div_survived_chart,
        script_survived_chart=script_survived_chart,
        div_title_chart=div_title_chart,
        script_title_chart=script_title_chart,
        div_hist_age=div_hist_age,
        script_hist_age=script_hist_age,
        selected_class=selected_class
    )

################################################################################
#                            - END OF MAIN ROUTE -                             #
################################################################################


################################################################################
#                        > CHART GENERATION FUNCTIONS                          #
################################################################################

def survived_bar_chart(dataset, patient_stage, cpalette=palette[1:4]):
    surv_data = dataset[dataset['stage'] == int(patient_stage)]
    surv_possibilities = list(surv_data['survival_5_years'].value_counts().index)
    surv_values = list(surv_data['survival_5_years'].value_counts().values)
    surv_possibilities_text = ['Did not Survive', 'survival_5_years']
        
    source = ColumnDataSource(data={
        'possibilities': surv_possibilities,
        'possibilities_txt': surv_possibilities_text,
        'values': surv_values
    })

    hover_tool = HoverTool(
        tooltips=[('survival_5_years?', '@possibilities_txt'), ('Count', '@values')]
    )
    
    p = figure(tools=[hover_tool], plot_height=400, title='Did/Did not Survive for Current stage')
    p.vbar(x='possibilities', top='values', source=source, width=0.9,
           fill_color=factor_cmap('possibilities_txt', palette=palette_generator(len(source.data['possibilities_txt']), cpalette), factors=source.data['possibilities_txt']))
    
    plot_styler(p)
    p.xaxis.ticker = source.data['possibilities']
    p.xaxis.major_label_overrides = { 0: 'Did not Survive', 1: 'Survived' }
    p.sizing_mode = 'scale_width'
    
    return p


def class_titles_bar_chart(dataset, patient_stage, cpalette=palette):
    ttl_data = dataset[dataset['stage'] == int(patient_stage)]
    title_possibilities = list(ttl_data['province'].value_counts().index)
    title_values = list(ttl_data['province'].value_counts().values)
    int_possibilities = np.arange(len(title_possibilities))
    
    source = ColumnDataSource(data={
        'provinces': title_possibilities,
        'provinces_int': int_possibilities,
        'values': title_values
    })

    hover_tool = HoverTool(
        tooltips=[('province', '@provinces'), ('Count', '@values')]
    )
    
    chart_labels = {}
    for val1, val2 in zip(source.data['provinces_int'], source.data['provinces']):
        chart_labels.update({ int(val1): str(val2) })
        
    p = figure(tools=[hover_tool], plot_height=300, title='Provinces for Current stage')
    p.vbar(x='provinces_int', top='values', source=source, width=0.9,
           fill_color=factor_cmap('provinces', palette=palette_generator(len(source.data['provinces']), cpalette), factors=source.data['provinces']))
    
    plot_styler(p)
    p.xaxis.ticker = source.data['provinces_int']
    p.xaxis.major_label_overrides = chart_labels
    p.xaxis.major_label_orientation = math.pi / 4
    p.sizing_mode = 'scale_width'
    
    return p


def age_hist(dataset, patient_stage, color=palette[1]):
    hist, edges = np.histogram(dataset[dataset['stage'] == int(patient_stage)]['age'].fillna(df['age'].mean()), bins=25)
    
    source = ColumnDataSource({
        'hist': hist,
        'edges_left': edges[:-1],
        'edges_right': edges[1:]
    })

    hover_tool = HoverTool(
        tooltips=[('From', '@edges_left'), ('Thru', '@edges_right'), ('Count', '@hist')], 
        mode='vline'
    )
    
    p = figure(plot_height=400, title='Age Histogram', tools=[hover_tool])
    p.quad(top='hist', bottom=0, left='edges_left', right='edges_right', source=source,
            fill_color=color, line_color='black')

    plot_styler(p)
    p.sizing_mode = 'scale_width'

    return p

################################################################################
#                    - END OF CHART GENERATION FUNCTIONS -                     #
################################################################################
#-----------------end of cervical cancer visualisations------------------------------------------
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#prostate cancer visualisations
@app.route('/chart_prostate', methods=['GET', 'POST'])
def chart_prostate():
    selected_class = request.form.get('dropdown-select')

    if selected_class == 0 or selected_class == None:
        survived_chart, title_chart, hist_age = redraw(1)
    else:
        survived_chart, title_chart, hist_age = redraw(selected_class)

    script_survived_chart, div_survived_chart = components(survived_chart)
    script_title_chart, div_title_chart = components(title_chart)
    script_hist_age, div_hist_age = components(hist_age)

    return render_template(
        'prostate_visu_index.html',
        div_survived_chart=div_survived_chart,
        script_survived_chart=script_survived_chart,
        div_title_chart=div_title_chart,
        script_title_chart=script_title_chart,
        div_hist_age=div_hist_age,
        script_hist_age=script_hist_age,
        selected_class=selected_class
    )

################################################################################
#                            - END OF MAIN ROUTE -                             #
################################################################################


################################################################################
#                        > CHART GENERATION FUNCTIONS                          #
################################################################################

def survived_bar_chart(dataset, patient_stage, cpalette=palette[1:4]):
    surv_data = dataset[dataset['stage'] == int(patient_stage)]
    surv_possibilities = list(surv_data['survival_5_years'].value_counts().index)
    surv_values = list(surv_data['survival_5_years'].value_counts().values)
    surv_possibilities_text = ['Did not Survive', 'survival_5_years']
        
    source = ColumnDataSource(data={
        'possibilities': surv_possibilities,
        'possibilities_txt': surv_possibilities_text,
        'values': surv_values
    })

    hover_tool = HoverTool(
        tooltips=[('survival_5_years?', '@possibilities_txt'), ('Count', '@values')]
    )
    
    p = figure(tools=[hover_tool], plot_height=400, title='Did/Did not Survive for Current stage')
    p.vbar(x='possibilities', top='values', source=source, width=0.9,
           fill_color=factor_cmap('possibilities_txt', palette=palette_generator(len(source.data['possibilities_txt']), cpalette), factors=source.data['possibilities_txt']))
    
    plot_styler(p)
    p.xaxis.ticker = source.data['possibilities']
    p.xaxis.major_label_overrides = { 0: 'Did not Survive', 1: 'Survived' }
    p.sizing_mode = 'scale_width'
    
    return p


def class_titles_bar_chart(dataset, patient_stage, cpalette=palette):
    ttl_data = dataset[dataset['stage'] == int(patient_stage)]
    title_possibilities = list(ttl_data['race'].value_counts().index)
    title_values = list(ttl_data['race'].value_counts().values)
    int_possibilities = np.arange(len(title_possibilities))
    
    source = ColumnDataSource(data={
        'races': title_possibilities,
        'races_int': int_possibilities,
        'values': title_values
    })

    hover_tool = HoverTool(
        tooltips=[('race', '@races'), ('Count', '@values')]
    )
    
    chart_labels = {}
    for val1, val2 in zip(source.data['races_int'], source.data['races']):
        chart_labels.update({ int(val1): str(val2) })
        
    p = figure(tools=[hover_tool], plot_height=300, title='Races for Current stage')
    p.vbar(x='races_int', top='values', source=source, width=0.9,
           fill_color=factor_cmap('races', palette=palette_generator(len(source.data['races']), cpalette), factors=source.data['races']))
    
    plot_styler(p)
    p.xaxis.ticker = source.data['races_int']
    p.xaxis.major_label_overrides = chart_labels
    p.xaxis.major_label_orientation = math.pi / 4
    p.sizing_mode = 'scale_width'
    
    return p


def age_hist(dataset, patient_stage, color=palette[1]):
    hist, edges = np.histogram(dataset[dataset['stage'] == int(patient_stage)]['age'].fillna(df1['age'].mean()), bins=25)
    
    source = ColumnDataSource({
        'hist': hist,
        'edges_left': edges[:-1],
        'edges_right': edges[1:]
    })

    hover_tool = HoverTool(
        tooltips=[('From', '@edges_left'), ('Thru', '@edges_right'), ('Count', '@hist')], 
        mode='vline'
    )
    
    p = figure(plot_height=400, title='Age Histogram', tools=[hover_tool])
    p.quad(top='hist', bottom=0, left='edges_left', right='edges_right', source=source,
            fill_color=color, line_color='black')

    plot_styler(p)
    p.sizing_mode = 'scale_width'

    return p

################################################################################
#                    - END OF CHART GENERATION FUNCTIONS -                     #
################################################################################

# call model to predict an image
def prostate_detection_api(full_path):
    data = image.load_img(full_path, target_size=(150, 150, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255

    with graph.as_default():
        predicted = model5.predict(data)
        return predicted


# home page
@app.route('/prostate_detection')
def prostate_detection():
   return render_template('prostate_detection_index.html')


# procesing uploaded file and predict it
@app.route('/upload_prostate', methods=['POST','GET'])
def upload_file_prostate():

    if request.method == 'GET':
        return render_template('prostate_detection_index.html')
    else:
        file = request.files['image']
        full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(full_name)

        indices = {0: 'cancer', 1: 'Not cancer'}
        result = prostate_detection_api(full_name)

        predicted_class = np.asscalar(np.argmax(result, axis=1))
        accuracy = round(result[0][predicted_class] * 100, 2)
        label = indices[predicted_class]

    return render_template('prostate_detection_predict.html', image_file_name = file.filename, label = label, accuracy = accuracy)


@app.route('/uploads/<filename>')
def prostate_send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

######################################CERVICAL DETECTION#############################################################
#####################################################################################################################
#####################################################################################################################
# call model to predict an image
def cervical_detection_api(full_path):
    data1 = image.load_img(full_path, target_size=(256, 256))
    
    data = image.img_to_array(data1)
    data = np.expand_dims(data, axis=0) * 1./255
   

    with graph.as_default():
        predicted = model6.predict(data)
        return predicted


# home page
@app.route('/cervical_detection')
def cervical_detection():
   return render_template('cervical_detection_index.html')


# procesing uploaded file and predict it
@app.route('/upload_cervical', methods=['POST','GET'])
def cervical_upload_file():

    if request.method == 'GET':
        return render_template('cervical_detection_index.html')
    else:
        file = request.files['image']
        full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(full_name)

        indices = {0: 'cancer', 1: 'Not cancer'}
        result = cervical_detection_api(full_name)

        predicted_class = np.asscalar(np.argmax(result, axis=1))
        accuracy = round(result[0][predicted_class] * 100, 2)
        label = indices[predicted_class]

    return render_template('cervical_detection_predict.html', image_file_name = file.filename, label = label, accuracy = accuracy)


@app.route('/uploads/<filename>')
def cervical_send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    app.run(port=8080, debug=True)






