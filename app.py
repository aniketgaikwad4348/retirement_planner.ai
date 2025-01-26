from flask import Flask, request, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_migrate import Migrate
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer, SignatureExpired
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import os
from statsmodels.tsa.arima.model import ARIMA
from werkzeug.utils import secure_filename
import wikipediaapi
import logging
import joblib
import json
from flask import Flask, request, render_template, jsonify
from flask import session



# Initialize the Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'your_email@gmail.com'
app.config['MAIL_PASSWORD'] = 'your_password'
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['UPLOAD_FOLDER'] = 'static/profile_photos'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
migrate = Migrate(app, db)
mail = Mail(app)
s = URLSafeTimedSerializer(app.config['SECRET_KEY'])

# User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    firstname = db.Column(db.String(100), nullable=False)
    middlename = db.Column(db.String(100), nullable=True)
    lastname = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    profile_photo_url = db.Column(db.String(200), nullable=True)
    is_admin = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f"User('{self.firstname}', '{self.lastname}', '{self.email}')"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))






def suggest_top_investments(user_input):
    try:
        # Extract relevant features from the user input
        user_input_df = pd.DataFrame([user_input])

        # Make sure all required features are in the user input
        #user_input_encoded_growth = preprocess_user_input(user_input, growth_model_features)
        #user_input_encoded_trend = preprocess_user_input(user_input, trend_model_features)

        # Predict expected growth and trend score for each investment type
        predictions = {}
        for investment_type in ['stocks', 'bonds', 'mutual fund', 'real estate']:
            # Load corresponding CSV data
            csv_file = f'{investment_type.replace(" ", "_")}_data.csv'
            data = pd.read_csv(csv_file)

            if data.empty:
                logging.debug(f"No data available for investment type '{investment_type}'.")
                continue

            # Predict expected growth and trend score for each investment
            #data['expected_growth'] = growth_model.predict(preprocess_user_input(data.to_dict(orient='records'), growth_model_features))
            #data['trend_score'] = trend_model.predict(preprocess_user_input(data.to_dict(orient='records'), trend_model_features))

            # Filter and sort investments
            filtered_investments = data.drop_duplicates(subset=['symbol'])
            top_investments = filtered_investments[['symbol', 'expected_growth', 'trend_score', 'company_name']].sort_values(
                by=['expected_growth', 'trend_score'], ascending=[False, False]).head(3)

            # Store predictions
            predictions[investment_type] = top_investments.to_dict(orient='records')

        logging.debug(f"Top investments for each type: {predictions}")

        return predictions

    except Exception as e:
        logging.error(f"Error in suggest_top_investments: {e}")
        return {"message": "An error occurred while suggesting investments."}



# Routes





@app.route('/features')
def features():
    return render_template('features.html')


from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import numpy as np
import pickle
import os

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'desired_retirement_income_model.pkl')

try:
    model = joblib.load(model_path)
    if not hasattr(model, 'predict'):
        raise AttributeError("Loaded model does not have a 'predict' method.")
except Exception as e:
    print(f"An error occurred while loading the model: {str(e)}")
    model = None

@app.route('/income_prediction_form', methods=['GET'])
@login_required
def income_prediction_form():
    return render_template('income_prediction_form.html')


# predict_income route
@app.route('/predict_income', methods=['POST'])
@login_required
def predict_income():
    if model is None:
        return jsonify({'error': 'Model is not loaded or is not valid.'}), 500

    try:
        # Initialize warnings
        warnings_list = []

        # Extract form data
        age = int(request.form.get('age', '0'))
        retirement_age = int(request.form.get('retirement_age', '0'))
        current_savings = float(request.form.get('current_savings', '0'))
        income = float(request.form.get('income', '0'))
        investment_account_balance = float(request.form.get('investment_account_balance', '0'))
        monthly_savings = float(request.form.get('monthly_savings', '0'))
        expected_investment_return_rate = float(request.form.get('expected_investment_return_rate', '0'))

        # Calculate total current savings
        total_current_savings = current_savings + investment_account_balance

        # Validate inputs
        if not (18 <= age <= 69):
            warnings_list.append(f"Age {age} is outside the expected range (18-69).")
        if not (age < retirement_age <= 100):
            warnings_list.append(f"Retirement age {retirement_age} must be greater than current age {age} and less than or equal to 100.")
        if not (14937 <= income <= 105012):
            warnings_list.append(f"Income {income} is outside the expected range ($14,937 - $105,012).")
        if not (-1368 <= total_current_savings <= 95320):
            warnings_list.append(f"Total current savings {total_current_savings} is outside the expected range (-$1,368 - $95,320).")
        if not (-5000 <= investment_account_balance <= 25004):
            warnings_list.append(f"Investment account balance {investment_account_balance} is outside the expected range (-$5,000 - $25,004).")
        if not (-400 <= monthly_savings <= 1398):
            warnings_list.append(f"Monthly savings {monthly_savings} is outside the expected range (-$400 - $1,398).")
        if not (0.02 <= expected_investment_return_rate <= 0.07):
            warnings_list.append(f"Expected investment return rate {expected_investment_return_rate} is outside the expected range (2% - 7%).")

        # If there are any warnings, return to the input form with warnings
        if warnings_list:
            return render_template('input_form.html', warnings=warnings_list)

        # Create an input array for the model
        input_features = np.array([[age, income, total_current_savings, investment_account_balance, 
                                    monthly_savings, retirement_age, expected_investment_return_rate]])

        # Make a prediction using the loaded model
        predicted_income = model.predict(input_features)

        # Store values in session for later use
        session['age'] = age
        session['retirement_age'] = retirement_age
        session['total_current_savings'] = total_current_savings  # Store the total current savings
        session['income'] = income
        session['predicted_income'] = predicted_income[0]
        session['expected_investment_return_rate'] = expected_investment_return_rate
        session['monthly_savings'] = monthly_savings  # Store monthly savings

        # Render the result with any warnings
        return render_template('income_prediction_summary.html', 
                               age=age, 
                               retirement_age=retirement_age, 
                               total_current_savings=total_current_savings,  
                               income=income,  
                               predicted_income=predicted_income[0], 
                               expected_investment_return_rate=expected_investment_return_rate, 
                               monthly_savings=monthly_savings,  # Pass monthly savings to the template
                               warnings=warnings_list)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction. Please try again.'}), 500


# planning route
@app.route('/planning', methods=['POST', 'GET'])
@login_required
def planning():
    if request.method == 'POST':
        # Extract and store form data in session
        age = request.form.get('age', 0, type=int)
        retirement_age = request.form.get('retirement_age', 0, type=int)
        total_current_savings = request.form.get('total_current_savings', 0.0, type=float)
        monthly_savings = request.form.get('monthly_savings', 0.0, type=float)
        income = request.form.get('income', 0.0, type=float)
        predicted_income = request.form.get('predicted_income', 0.0, type=float)
        expected_investment_return_rate = request.form.get('expected_investment_return_rate', 0.0, type=float)

        # Store values in session
        session['age'] = age
        session['retirement_age'] = retirement_age
        session['total_current_savings'] = total_current_savings
        session['income'] = income
        session['predicted_income'] = predicted_income
        session['expected_investment_return_rate'] = expected_investment_return_rate
        session['monthly_savings'] = monthly_savings
        print(f"Debug: Current Monthly Savings stored in session: {session['monthly_savings']}")

    else:
        # Retrieve from session if no POST data
        age = session.get('age', 0)
        retirement_age = session.get('retirement_age', 0)
        total_current_savings = session.get('total_current_savings', 0.0)
        income = session.get('income', 0.0)
        predicted_income = session.get('predicted_income', 0.0)
        monthly_savings = session.get('monthly_savings', 0.0)  # Retrieve monthly savings from session
        expected_investment_return_rate = session.get('expected_investment_return_rate', 0.0)

    app.logger.debug(f"Debug: Age: {age}, Retirement Age: {retirement_age}")

    # Validate inputs
    if age == 0 or retirement_age == 0 or income <= 0 or predicted_income <= 0 or expected_investment_return_rate <= 0:
        flash('Please provide valid inputs.', 'error')
        return redirect(url_for('planning'))

    # Calculate savings
    savings_data = calculate_savings(
        age, retirement_age, total_current_savings, predicted_income, expected_investment_return_rate, 85
    )
    if not savings_data:
        flash('Calculation error occurred', 'error')
        return redirect(url_for('planning'))

    # Calculate monthly savings needed
    annual_savings_needed = savings_data['annual_savings_needed']
    monthly_savings_needed = annual_savings_needed / 12

    # Render template with all necessary values
    return render_template('planning.html', 
                           current_monthly_savings=session.get('monthly_savings', 0.0),  # Pass stored value from session
                           monthly_savings_needed=monthly_savings_needed,      # Pass calculated value
                           retirement_age=retirement_age,
                           age=age,
                           annual_savings_needed=annual_savings_needed,
                           present_value_required_income=savings_data['present_value_required_income'],
                           total_income_needed=savings_data['total_income_needed'],
                           total_savings_needed=savings_data['total_savings_needed'],
                           total_current_savings=session.get('total_current_savings', 0.0),
                           predicted_income=session.get('predicted_income', 0.0))

# Define your savings calculation function
def calculate_savings(age, retirement_age, total_current_savings, expected_retirement_income, expected_investment_return_rate, life_expectancy):
    # Calculate years to retirement
    years_to_retirement = retirement_age - age
    retirement_duration = life_expectancy - retirement_age

    # Total income needed during retirement
    total_income_needed = expected_retirement_income * retirement_duration

    # Calculate present value of required retirement income
    r = expected_investment_return_rate / 100  # Convert percentage to decimal
    pv_required_income = (expected_retirement_income * (1 - (1 + r) ** -retirement_duration) / r) if r > 0 else expected_retirement_income * retirement_duration

    # Total savings needed
    total_savings_needed = pv_required_income - total_current_savings

    # Annual savings needed
    annual_savings_needed = total_savings_needed / years_to_retirement if years_to_retirement > 0 else 0

    return {
        "total_income_needed": total_income_needed,
        "present_value_required_income": pv_required_income,
        "total_savings_needed": total_savings_needed,
        "annual_savings_needed": annual_savings_needed
    }

@app.route('/calculate_required_savings', methods=['POST'])
@login_required
def calculate_required_savings():
    # Retrieve data from session
    age = session.get('age', 0)
    retirement_age = session.get('retirement_age', 0)
    total_current_savings = session.get('total_current_savings', 0.0)
    income = session.get('income', 0.0)
    predicted_income = session.get('predicted_income', 0.0)
    expected_investment_return_rate = session.get('expected_investment_return_rate', 0.0)

    # Debugging lines
    print(f"Debug: Age: {age}, Retirement Age: {retirement_age}, Total Current Savings: {total_current_savings}, Income: {income}, Predicted Income: {predicted_income}, Expected Return Rate: {expected_investment_return_rate}")

    # Validate inputs
    if age == 0 or retirement_age == 0 or total_current_savings < 0 or predicted_income <= 0 or expected_investment_return_rate <= 0:
        return jsonify({'error': 'Please provide valid inputs.'}), 400

    # Call the savings calculation function
    savings_data = calculate_savings(age, retirement_age, total_current_savings, predicted_income, expected_investment_return_rate, 85)  # Assuming default life expectancy

    # Pass the savings data to the template
    return render_template('calculate_required_savings.html', 
                           annual_savings_needed=savings_data['annual_savings_needed'],
                           present_value_required_income=savings_data['present_value_required_income'],
                           total_income_needed=savings_data['total_income_needed'],
                           total_savings_needed=savings_data['total_savings_needed'])

@app.route('/life_expectancy_form', methods=['GET'])
@login_required
def life_expectancy_form():
    age = session.get('age', 0)
    retirement_age = session.get('retirement_age', 0)
    total_current_savings = session.get('total_current_savings', 0.0)
    income = session.get('income', 0.0)
    predicted_income = session.get('predicted_income', 0.0)
    expected_investment_return_rate = session.get('expected_investment_return_rate', 0.0)

    print(f"Debug: Total Current Savings from session: {total_current_savings}")  # Debugging line

    return render_template('life_expectancy_form.html', 
                           age=age, 
                           retirement_age=retirement_age, 
                           total_current_savings=total_current_savings, 
                           income=income, 
                           predicted_income=predicted_income, 
                           expected_investment_return_rate=expected_investment_return_rate)






@app.route('/predict_saving_form')
def predict_saving_form():
    return render_template('predict_saving_form.html')

@app.route('/predict_inflation_market')
def predict_inflation_market():
    # Logic for predicting inflation and market return
    return "Predict Inflation & Market Return Page"

@app.route('/predict_retirement_goal')
def predict_retirement_goal():
    # Logic for predicting retirement goal
    return "Predict Retirement Goal Page"

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        firstname = request.form['firstname']
        middlename = request.form['middlename']
        lastname = request.form['lastname']
        email = request.form['email']
        password = request.form['password']

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('This email is already registered. Please use a different email or log in.', 'danger')
            return redirect(url_for('register'))

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        new_user = User(firstname=firstname, middlename=middlename, lastname=lastname, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Your account has been created! You can now log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Login unsuccessful. Please check your email and password.', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        firstname = request.form.get('firstname')
        middlename = request.form.get('middlename')
        lastname = request.form.get('lastname')
        email = request.form.get('email')

        if 'profile_photo' in request.files:
            profile_photo = request.files['profile_photo']
            if profile_photo:
                filename = secure_filename(profile_photo.filename)
                photo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                profile_photo.save(photo_path)
                current_user.profile_photo_url = photo_path

        current_user.firstname = firstname
        current_user.middlename = middlename
        current_user.lastname = lastname
        current_user.email = email

        db.session.commit()
        flash('Your profile has been updated!', 'success')
        return redirect(url_for('profile'))

    return render_template('profile.html', user=current_user)

@app.route('/update_user/<int:user_id>', methods=['POST'])
@login_required
def update_user(user_id):
    if not current_user.is_admin:
        flash('You are not authorized to perform this action.', 'danger')
        return redirect(url_for('admin_dashboard'))

    user = User.query.get_or_404(user_id)
    user.firstname = request.form['firstname']
    user.middlename = request.form['middlename']
    user.lastname = request.form['lastname']
    user.email = request.form['email']
    user.is_admin = 'is_admin' in request.form

    db.session.commit()
    flash('User details have been updated!', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/edit_user_admin/<int:id>', methods=['GET', 'POST'])
@login_required
def edit_user_admin(id):
    if not current_user.is_admin:
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('home'))

    user = User.query.get_or_404(id)

    if request.method == 'POST':
        user.firstname = request.form['firstname']
        user.middlename = request.form['middlename']
        user.lastname = request.form['lastname']
        user.email = request.form['email']
        user.is_admin = 'is_admin' in request.form

        db.session.commit()
        flash('User details have been updated!', 'success')
        return redirect(url_for('admin_dashboard'))

    return render_template('edit_user_admin.html', user=user)



# Initialize Wikipedia API with a user agent
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='YourAppName/1.0 (your_email@example.com)'
)



@app.route('/investment_details/<investment_symbol>')
@login_required
def investment_details(investment_symbol):
    try:
        top_performing_companies = pd.read_csv('cleaned_top_performing_companies.csv')
    except Exception as e:
        return render_template('error.html', message="Unable to load investment data.")
    
    investment_details = top_performing_companies[top_performing_companies['symbol'] == investment_symbol]
    
    if investment_details.empty:
        flash('No details found for this investment.', 'warning')
        return redirect(url_for('home'))

    investment_details = investment_details.iloc[0]
    investment_details_dict = investment_details.to_dict()

    # Ensure NaN values are handled
    investment_details_dict['expected_growth'] = investment_details_dict.get('expected_growth', 'N/A')
    investment_details_dict['trend_score'] = investment_details_dict.get('trend_score', 'N/A')
    investment_details_dict['moving_avg'] = investment_details_dict.get('moving_avg', 'N/A')

    # Fetch Wikipedia information
    try:
        page = wiki_wiki.page(investment_details_dict.get('company', ''))
        additional_info_wikipedia = page.summary if page.exists() else 'No additional information available from Wikipedia.'
    except Exception as e:
        additional_info_wikipedia = 'No additional information available from Wikipedia.'

    
    # Extract historical data for the graph
    historical_data = top_performing_companies[top_performing_companies['symbol'] == investment_symbol].sort_values(by='timestamp')
    dates = historical_data['timestamp'].tolist()
    open_prices = historical_data['open'].tolist()

    return render_template('investment_details.html', 
                           investment=investment_details_dict,
                           dates=dates,
                           open_prices=open_prices)


@app.route('/admin')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash('You are not authorized to view this page.', 'danger')
        return redirect(url_for('index'))

    users = User.query.all()  # Fetch all users
    return render_template('admin.html', users=users)


@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    # Check if the current user is an admin (you might have a role check here)
    if not current_user.is_admin:
        flash('You are not authorized to perform this action.', 'danger')
        return redirect(url_for('admin_dashboard'))

    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    flash('User has been deleted successfully.', 'success')
    return redirect(url_for('admin_dashboard'))




@app.route('/upload_photo', methods=['POST'])
@login_required
def upload_photo():
    if 'profile_photo' in request.files:
        profile_photo = request.files['profile_photo']
        filename = secure_filename(profile_photo.filename)
        photo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        profile_photo.save(photo_path)
        
        # Update the user's profile_photo_url to point to the static folder
        current_user.profile_photo_url = url_for('static', filename=f'profile_photos/{filename}')
        db.session.commit()
        
        flash('Profile photo updated!', 'success')
    else:
        flash('No photo selected.', 'error')
    
    return redirect(url_for('profile'))



# Run the app
if __name__ == '__main__':
    app.run(debug=True)
    test_types = ['stocks', 'bonds', 'real_estate', 'mutual_funds']
    results = {t: suggest_top_investments(t) for t in test_types}
    print(results)
