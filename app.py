from flask import Flask, render_template, request, url_for, flash, redirect
from markupsafe import escape

app = Flask(__name__)
app.secret_key = '7570fe04242b23c01ba09d1a0a7152ec59a35ef29008f54c3aa5632b8ef47fec'

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