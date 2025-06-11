from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:5342@localhost:5432/postgres'
db = SQLAlchemy(app)

with app.app_context():
    # Execute SQL to list all databases
    result = db.session.execute(text("SELECT datname FROM pg_database WHERE datistemplate = false;"))
    databases = [row[0] for row in result]
    print("Available databases:")
    for database in databases:
        print(f"- {database}")
