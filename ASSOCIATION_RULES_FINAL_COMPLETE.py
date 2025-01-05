"""

1.	Business Problem
1.1.	What is the business objective?
1.2.	Are there any constraints?
2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image:


3.	Data Pre-processing
3.1	Data Cleaning, Feature Engineering, etc.
4.	Model Building
4.1 Application of Apriori Algorithm
4.2	Build the most frequent item sets and plot the rules
4.3	Work on Codes
5. Deployment
5.1 Deploy solutions
6. Write about the benefits/impact of the solution - in what way does the business (client) benefit from the solution provided?


"""


'''

Problem Statement: -
Kitabi Duniya, a famous bookstore in India, was established before Independence, the growth of the company was incremental year by year,
but due to the online selling of books and widespread Internet access, its annual growth started to collapse. As a Data Scientist,
you must help this heritage bookstore gain its popularity back and increase the footfall of customers and 
provide ways to improve the business exponentially to an expected value at a 25% improvement of the current rate. Apply the pattern mining techniques 
(Association Rules Algorithm) to identify ways to improve sales. Explain the rules (patterns) identified,
and visually represent the rules in graphs for a clear understanding of the solution.


 

Business Objective: maximize the footfall of customers.

Business Constraints: with the limited budget and resources.

Success Criteria: 
	
    Business Success Criteria: Increase in Sales by 25%.
	ML Success Criteria: apriori algorithm will achieve an accuracy of 90%.
	Economic Success Criteria: bookstore will see an increase in revenue by 25%. 


'''

# Data Dictionary


'''

ChildBks : Indicates if a customer purchased Children's Books
YouthBks : 	Indicates if a customer purchased Youth Books
CookBks : 	Indicates if a customer purchased Cook Books
DoItYBks : Indicates if a customer purchased Do-It-Yourself Books
RefBks : Indicates if a customer purchased Reference Books
ArtBks : Indicates if a customer purchased Art Books
GeogBks : Indicates if a customer purchased Geography Books
ItalCook : Indicates if a customer purchased Italian Cook Books
ItalAtlas : 	Indicates if a customer purchased Italian Atlases
ItalArt : Indicates if a customer purchased Italian Art Books
Florence : Indicates if a customer purchased Books related to Florence

'''





# Importing necessary libraries for data manipulation and visualization
import pandas as pd
from sqlalchemy import create_engine, text  # For SQL Database connection
from urllib.parse import quote  # For parsing the password in the database URL
import matplotlib.pyplot as plt  # For plotting graphs
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
import numpy as np
import pickle

# Reading the CSV file containing transaction data into a DataFrame
book = pd.read_csv('C:/Users/user/Desktop/DATA SCIENCE SUBMISSION/Data Science [ 5 ] - Association Rule/book.csv')

# Displaying the first few rows of the DataFrame to understand the data structure
book.head()

# Getting additional information about the DataFrame's columns and data types
book.info()

# Setting up a connection to a SQL Database using SQLAlchemy
# 'user', 'pw', and 'db' are placeholders for the database user, password, and database name
user = 'root'
pw = '12345678'
db = 'univ_db'

# Creating a SQLAlchemy engine for connecting to the MySQL database
engine = create_engine(f"mysql+pymysql://{user}:{quote(pw)}@localhost/{db}")

# Loading the DataFrame into the MySQL database, replacing the table if it already exists
book.to_sql('association_rule_table', con=engine, if_exists='replace', chunksize=1000, index=False)

# Reading the data back from the SQL database into a new DataFrame
sql = text('select * from association_rule;')
book_1 = pd.read_sql_query(sql, con=engine.connect())


# Examining the first 100 rows of the data to ensure it loaded correctly
book_1.head(100)




# ANALYSIS EDA

# Summing the binary columns to count the number of purchases for each book type
count = book.loc[:, :].sum()

# Sorting the items based on the count to identify the most popular items
pop_item = count.sort_values(ascending=False).head(10)

# Converting the resulting series into a DataFrame for easier manipulation
pop_item = pop_item.to_frame()

# Resetting the index of the DataFrame to have a clean table format
pop_item = pop_item.reset_index()

# Renaming the DataFrame columns to more descriptive names
pop_item = pop_item.rename(columns={"index": "items", 0: "count"})


# Setting a default size for all plots, making them clearer and consistent
plt.rcParams['figure.figsize'] = (10, 6)


# Data Visualization: Plotting a horizontal bar chart of the most popular items
plt.style.use('dark_background')  # Using a dark background for the plot for better contrast
pop_item.plot.barh()
plt.title('Most popular items')  # Setting the title of the plot
plt.gca().invert_yaxis()  # Inverting the y-axis to have the top item at the top






# Installing the mlxtend library (if needed) - This should be run in the command line or terminal
# pip install mlxtend

# Importing functions for frequent pattern mining and generating association rules
from mlxtend.frequent_patterns import apriori, association_rules

import warnings
warnings.filterwarnings('ignore')  # Ignoring any warnings for a cleaner output



# Applying the Apriori algorithm to find frequent itemsets in the dataset
# min_support specifies the minimum support of the itemsets to be found
frequent_itemset = apriori(book, min_support=0.0075, use_colnames=True, max_len=4)



# Displaying the top and bottom frequent itemsets
frequent_itemset.head(10)
frequent_itemset.tail(10)
frequent_itemset.info()



# Generating association rules from the frequent itemsets using lift as the metric
rules = association_rules(frequent_itemset, metric="lift", min_threshold=1)



# Displaying the top and bottom association rules
rules.head()
rules.tail()
rules.info()



# Function to sort itemsets alphabetically
def to_list(i):
    return list(sorted(i))

# Creating concatenated and sorted lists of antecedents and consequents for each rule
concat_list = rules['antecedents'].apply(to_list) + rules['consequents'].apply(to_list)
concat_list1 = concat_list.apply(sorted)



# Obtaining unique sets of rules by converting lists to tuples and back
rules_set = list(concat_list1)


unique_tuple = set(map(tuple, rules_set))  # Using a set to filter out duplicates
unique_rule_set = [list(i) for i in unique_tuple]



# Finding original index positions of the unique rules
unique_index = []
for i in unique_rule_set:
    unique_index.append(rules_set.index(i))



# Extracting rules without redundancy using the unique indices
rules_without_retundancy = rules.iloc[unique_index, :]



# Cleaning the rule items by removing unnecessary string parts for better readability
# database do not accept frozenset

rules['antecedents'] = rules['antecedents'].astype('string')
rules['consequents'] = rules['consequents'].astype('string')


rules['antecedents'] = rules['antecedents'].str.removeprefix("frozenset({")
rules['antecedents'] = rules['antecedents'].str.removesuffix("})")



rules['consequents'] = rules['consequents'].str.removeprefix("frozenset({")
rules['consequents'] = rules['consequents'].str.removesuffix("})")



# Sorting the rules by lift and selecting the top 10 for detailed analysis
rules10 = rules.sort_values('lift', ascending=False).head(10)



# Scatter plot of the top 10 rules based on support and confidence, colored by lift
rules10.plot(x="support", y="confidence", c=rules10.lift, kind="scatter", s=12, cmap=plt.cm.coolwarm)


# Replace -inf and inf values with NaN in the DataFrame rules10
rules10.replace([np.inf, -np.inf], np.nan, inplace=True)



# deploying the solution in the sql and csv file.

# Writing the top 10 strong rules back to the database for future analysis
rules10.to_sql('association_rule_top10', con=engine, if_exists='replace', chunksize=1000, index=False)


# loading the rules10 into the csv file.
rules10.to_csv('association_rule_top10.csv',encoding = 'utf-8' ,index = False )


 











# Benefits and Impact of the Solution for Kitabi Duniya:


"""

1. Identifying High-Value Association Rules:
Rule Example: {"ItalArt", "RefBks"} -> {"ItalAtlas", "ArtBks"}
Support: 0.0165 | Confidence: 0.825 | Lift: 45.83
Interpretation: Customers who buy Italian art books ("ItalArt") and reference books ("RefBks") are highly likely to also purchase Italian atlases ("ItalAtlas") and art books ("ArtBks"). The high lift (45.83) indicates that this is a strong and significant association.
Action: Kitabi Duniya could create a bundled offer featuring these items or place these items close to each other in the store to encourage combined purchases. This targeted promotion could lead to increased sales in these categories.


2. Targeted Marketing Campaigns:
Rule Example: {"ItalAtlas", "ItalCook"} -> {"ItalArt", "RefBks"}
Support: 0.0125 | Confidence: 0.543 | Lift: 27.17
Interpretation: Customers purchasing Italian atlases and Italian cookbooks are somewhat likely to also be interested in Italian art and reference books. The lift value of 27.17 indicates a strong association.
Action: Kitabi Duniya can target customers who have previously purchased Italian cookbooks with personalized recommendations or discounts on Italian art and reference books. This approach can improve conversion rates and enhance customer satisfaction.


3. Improving Store Layout and Product Placement:
Rule Example: {"ItalArt", "RefBks"} -> {"ItalAtlas", "ChildBks"}
Support: 0.0145 | Confidence: 0.725 | Lift: 25.44
Interpretation: Customers who purchase Italian art and reference books are also likely to buy Italian atlases and children's books. The strong confidence (0.725) suggests a good likelihood of this combined purchase.
Action: Kitabi Duniya could place children’s books near art and reference books, especially those related to Italian themes. This strategic product placement could increase impulse purchases and overall sales.


4. Bundling Strategies:
Rule Example: {"ItalAtlas", "ChildBks"} -> {"ItalArt", "RefBks"}
Support: 0.0145 | Confidence: 0.508 | Lift: 25.44
Interpretation: There’s a fair chance that customers who purchase Italian atlases and children's books might also be interested in Italian art and reference books.
Action: Kitabi Duniya can create bundle deals or package these items together at a discounted rate, encouraging customers to buy more items per visit, thereby increasing the average transaction value.


5. Personalized Customer Engagement:
Rule Example: {"ItalArt", "RefBks"} -> {"ItalAtlas", "DoItYBks"}
Support: 0.0095 | Confidence: 0.475 | Lift: 25.0
Interpretation: Customers who buy Italian art and reference books are also somewhat likely to purchase Italian atlases and Do-It-Yourself (DIY) books.
Action: Use this insight to personalize customer engagement. Kitabi Duniya can send personalized emails or notifications to customers who bought art and reference books, suggesting DIY and atlas books as complementary purchases.

"""



