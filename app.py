from flask import Flask, render_template, request, url_for, flash, redirect
from markupsafe import escape
import json

app = Flask(__name__)

debug= True # Debug mode should be off if hosted on an external website

def get_value_from_json(json_file, key, sub_key=None):
   '''
   Function to read the json file for our app secret key
   '''
   try:
       with open(json_file) as f:
           data = json.load(f)
           if sub_key:
               return data[key][sub_key]
           else:
               return data[key]
   except Exception as e:
       print("Error: ", e)

# Getting the credentials for the session and database access
app.secret_key = get_value_from_json("venv/secrets.json", "flask", "SECRET_KEY")
config = get_value_from_json("venv/secrets.json", "mysql_connector")


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/login', methods=('GET', 'POST'))
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', False)
        password = request.form.get('password', False)

        if not password:
            error = 'password is required!'
        elif not username:
            error = 'Username is required!'
        else:
            if username.lower() == 'natuyuki' and password == 'flask23':
                flash('You were successfully logged in')
                return redirect(url_for('train'))
            else:
                error = 'Incorrect username/password'
        return render_template('login.html', error= error)
    else:
        return render_template('login.html')

@app.route('/<name>')
def welcome(name):
    return f'Welcome, {escape(name)}'

@app.route('/train')
def train():
    return render_template('train.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=True)