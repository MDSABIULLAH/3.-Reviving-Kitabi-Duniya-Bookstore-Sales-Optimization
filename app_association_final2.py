# app.py
from flask import Flask, render_template, request, redirect, url_for
from io import StringIO
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sqlalchemy import create_engine
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('error.html', error_message='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('error.html', error_message='No selected file')
        if file and file.filename.endswith('.csv'):
            db_user = request.form['db_user']
            db_password = request.form['db_password']
            db_name = request.form['db_name']
            
            try:
                # Read the file content
                file_content = file.read().decode('utf-8')
                results, success_message = process_file_content(file_content, db_user, db_password, db_name)
                return render_template('result.html', results=results, success_message=success_message)
            except Exception as e:
                return render_template('error.html', error_message=str(e))
    return render_template('index.html')

def process_file_content(file_content, db_user, db_password, db_name):
    # Read the CSV content
    df = pd.read_csv(StringIO(file_content))
    
    # Perform association rule mining
    frequent_itemsets = apriori(df, min_support=0.0075, use_colnames=True, max_len=4)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    
    # Clean up the rules
    rules['antecedents'] = rules['antecedents'].astype('string')
    rules['consequents'] = rules['consequents'].astype('string')
    rules['antecedents'] = rules['antecedents'].str.removeprefix("frozenset({").str.removesuffix("})")
    rules['consequents'] = rules['consequents'].str.removeprefix("frozenset({").str.removesuffix("})")
    
    # Sort rules by lift and get top 10
    rules10 = rules.sort_values('lift', ascending=False).head(10)
    rules10.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Select the required columns
    columns_to_keep = ['antecedents', 'consequents', 'antecedent support', 'consequent support', 
                       'support', 'confidence', 'lift', 'leverage', 'conviction', 'zhangs_metric']
    rules10 = rules10[columns_to_keep]
    
    # Store results in the database
    engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@localhost/{db_name}")
    rules10.to_sql('association_rule_top10', con=engine, if_exists='replace', index=False)
    
    success_message = "Data has been successfully processed and saved to the database."
    
    return rules10.to_dict('records'), success_message

if __name__ == '__main__':
    app.run(debug=True)
