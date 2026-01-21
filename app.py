import streamlit as st
import pandas as pd
import sqlite3
import re
from code_editor import code_editor

# Page configuration
st.set_page_config(
    page_title="SQL Visual Learning Tool",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 SQL Visual Learning Tool")
st.markdown("**Learn SQL by seeing how tables transform step-by-step**")

# Sample data - customers table
@st.cache_data
def load_customers_data():
    data = {
        'customer_id': [1, 2, 3, 4, 5, 6],
        'name': ['John', 'Maria', 'Yuki', 'Ahmed', 'Sofia', 'Chen'],
        'country': ['USA', 'Greece', 'Japan', 'Egypt', 'Spain', 'China'],
        'age': [25, 30, 28, 35, 22, 40]
    }
    return pd.DataFrame(data)

# Sample data - orders table
@st.cache_data
def load_orders_data():
    data = {
        'order_id': [101, 102, 103, 104, 105, 106, 107, 108],
        'customer_id': [1, 2, 2, 3, 4, 5, 5, 5],
        'product': ['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard', 'Mouse', 'Headphones', 'Webcam'],
        'amount': [1200, 800, 450, 300, 80, 25, 150, 90]
    }
    return pd.DataFrame(data)

# DATASET 2: INTERMEDIATE - Library System
@st.cache_data
def load_books_data():
    data = {
        'book_id': [1, 2, 3, 4, 5, 6, 7, 8],
        'title': ['Python Basics', 'SQL Mastery', 'Data Analysis', 'Web Dev', 'AI Intro', 'Statistics', 'Excel Pro', 'Business Analytics'],
        'author': ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis'],
        'category': ['Programming', 'Database', 'Data Science', 'Web', 'AI', 'Math', 'Office', 'Business'],
        'price': [29.99, 39.99, 49.99, 34.99, 59.99, 44.99, 24.99, 39.99]
    }
    return pd.DataFrame(data)

@st.cache_data
def load_borrowers_data():
    data = {
        'borrow_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'book_id': [1, 2, 2, 3, 4, 5, 6, 7, 1, 2],
        'student_name': ['Anna', 'Bob', 'Chris', 'Diana', 'Elena', 'Frank', 'Grace', 'Helen', 'Ivan', 'Julia'],
        'days_borrowed': [7, 14, 21, 3, 10, 30, 5, 12, 7, 15]
    }
    return pd.DataFrame(data)


# DATASET 3: ADVANCED - Employee Performance
@st.cache_data
def load_employees_data():
    data = {
        'employee_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack'],
        'department_id': [1, 1, 2, 2, 3, 3, 1, 2, 3, None],
        'salary': [50000, 55000, 60000, 52000, 65000, 48000, 70000, 45000, 58000, 40000],
        'hire_year': [2020, 2019, 2021, 2020, 2018, 2022, 2019, 2021, 2020, 2023]
    }
    return pd.DataFrame(data)

@st.cache_data
def load_departments_data():
    data = {
        'department_id': [1, 2, 3, 4],
        'department_name': ['Sales', 'Engineering', 'Marketing', 'HR'],
        'budget': [100000, 200000, 80000, 60000]
    }
    return pd.DataFrame(data)

# DATASET 4: EXPERT - Search Trends (Time-Series & Multi-dimensional)
@st.cache_data
def load_search_trends_data():
    """Mimics Google Trends data structure with time-series and geographic dimensions"""
    from datetime import datetime, timedelta
    import random
    random.seed(42)
    
    # Generate 14 days of data
    base_date = datetime(2026, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(14)]
    
    # Search terms
    terms = ['Python tutorial', 'SQL basics', 'Data analysis', 'Machine learning', 'Excel tips']
    
    # Geographic dimensions (country + region for granularity)
    locations = [
        ('US', 'United States', 'CA', 'California'),
        ('US', 'United States', 'NY', 'New York'),
        ('US', 'United States', 'TX', 'Texas'),
        ('UK', 'United Kingdom', 'EN', 'England'),
        ('UK', 'United Kingdom', 'SC', 'Scotland'),
        ('DE', 'Germany', 'BE', 'Berlin'),
        ('DE', 'Germany', 'BY', 'Bavaria'),
    ]
    
    data = []
    for date in dates:
        for term in terms:
            for country_code, country_name, region_code, region_name in locations:
                # Generate search interest score (0-100)
                base_score = random.randint(40, 95)
                if term == 'Python tutorial':
                    score = min(100, base_score + 10)
                elif term == 'Excel tips':
                    score = base_score
                else:
                    score = base_score + random.randint(-5, 5)
                
                data.append({
                    'search_date': date.strftime('%Y-%m-%d'),
                    'search_term': term,
                    'country_code': country_code,
                    'country_name': country_name,
                    'region_code': region_code,
                    'region_name': region_name,
                    'search_interest': score
                })
    
    return pd.DataFrame(data)


@st.cache_data
def load_term_categories_data():
    """Dimension table for search term metadata"""
    data = {
        'search_term': ['Python tutorial', 'SQL basics', 'Data analysis', 'Machine learning', 'Excel tips'],
        'category': ['Programming', 'Database', 'Analytics', 'AI/ML', 'Office Tools'],
        'difficulty': ['Beginner', 'Beginner', 'Intermediate', 'Advanced', 'Beginner']
    }
    return pd.DataFrame(data)

# DATASET 5: CTE LEARNING - Sales Analytics
@st.cache_data
def load_sales_data():
    """Sales transactions for CTE examples"""
    data = {
        'sale_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'product_id': [1, 2, 1, 3, 2, 4, 1, 3, 5, 2, 4, 5, 1, 3, 2],
        'region': ['North', 'South', 'North', 'East', 'West', 'North', 'South', 'East', 'West', 'North', 'South', 'East', 'West', 'North', 'South'],
        'sale_date': ['2026-01-01', '2026-01-01', '2026-01-02', '2026-01-02', '2026-01-03', '2026-01-03', '2026-01-04', '2026-01-04', '2026-01-05', '2026-01-05', '2026-01-06', '2026-01-06', '2026-01-07', '2026-01-07', '2026-01-08'],
        'quantity': [5, 3, 8, 2, 6, 4, 10, 3, 7, 5, 6, 4, 9, 5, 8],
        'revenue': [250.00, 90.00, 400.00, 140.00, 180.00, 320.00, 500.00, 210.00, 105.00, 150.00, 480.00, 60.00, 450.00, 350.00, 240.00]
    }
    return pd.DataFrame(data)

@st.cache_data
def load_products_data():
    """Product catalog for CTE examples"""
    data = {
        'product_id': [1, 2, 3, 4, 5],
        'product_name': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Webcam'],
        'category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Accessories'],
        'unit_price': [50.00, 30.00, 70.00, 80.00, 15.00]
    }
    return pd.DataFrame(data)

# Dataset configurations
DATASETS = {
    "Dataset 1: Customers & Orders (Basic)": {
        "table1_name": "customers",
        "table2_name": "orders",
        "table1_data": load_customers_data,
        "table2_data": load_orders_data,
        "join_key": "customer_id",
        "description": "Learn basic JOINs with customer orders",
        "examples": {
            "Example 1: INNER JOIN": """SELECT customers.name, orders.product, orders.amount 
FROM customers 
INNER JOIN orders ON customers.customer_id = orders.customer_id""",
            
            "Example 2: LEFT JOIN (all customers)": """SELECT customers.name, orders.product, orders.amount 
FROM customers 
LEFT JOIN orders ON customers.customer_id = orders.customer_id""",
            
            "Example 3: JOIN with WHERE": """SELECT customers.name, orders.product, orders.amount 
FROM customers 
INNER JOIN orders ON customers.customer_id = orders.customer_id 
WHERE orders.amount > 200""",
            
            "Example 4: JOIN with ORDER BY": """SELECT customers.name, orders.product, orders.amount 
FROM customers 
INNER JOIN orders ON customers.customer_id = orders.customer_id 
ORDER BY orders.amount DESC""",
            
            "Example 5: Count orders per customer": """SELECT customers.name, COUNT(orders.order_id) as total_orders 
FROM customers 
LEFT JOIN orders ON customers.customer_id = orders.customer_id 
GROUP BY customers.name""",
            
            "Example 6: Total spent per customer": """SELECT customers.name, SUM(orders.amount) as total_spent 
FROM customers 
INNER JOIN orders ON customers.customer_id = orders.customer_id 
GROUP BY customers.name 
ORDER BY total_spent DESC"""
        }
    },
    
    "Dataset 2: Books & Borrowers (Intermediate)": {
        "table1_name": "books",
        "table2_name": "borrowers",
        "table1_data": load_books_data,
        "table2_data": load_borrowers_data,
        "join_key": "book_id",
        "description": "Practice with library data - includes duplicate borrows",
        "examples": {
            "Example 1: Which books were borrowed?": """SELECT books.title, borrowers.student_name, borrowers.days_borrowed 
FROM books 
INNER JOIN borrowers ON books.book_id = borrowers.book_id""",
            
            "Example 2: All books (even not borrowed)": """SELECT books.title, borrowers.student_name 
FROM books 
LEFT JOIN borrowers ON books.book_id = borrowers.book_id""",
            
            "Example 3: Books borrowed over 10 days": """SELECT books.title, borrowers.student_name, borrowers.days_borrowed 
FROM books 
INNER JOIN borrowers ON books.book_id = borrowers.book_id 
WHERE borrowers.days_borrowed > 10""",
            
            "Example 4: Most popular books": """SELECT books.title, COUNT(borrowers.borrow_id) as times_borrowed 
FROM books 
LEFT JOIN borrowers ON books.book_id = borrowers.book_id 
GROUP BY books.title 
ORDER BY times_borrowed DESC""",
            
            "Example 5: Category with most borrows": """SELECT books.category, COUNT(borrowers.borrow_id) as total_borrows 
FROM books 
LEFT JOIN borrowers ON books.book_id = borrowers.book_id 
GROUP BY books.category"""
        }
    },
    
    "Dataset 3: Employees & Departments (Advanced)": {
        "table1_name": "employees",
        "table2_name": "departments",
        "table1_data": load_employees_data,
        "table2_data": load_departments_data,
        "join_key": "department_id",
        "description": "Advanced: Includes NULL values and aggregations",
        "examples": {
            "Example 1: Employee departments": """SELECT employees.name, departments.department_name, employees.salary 
FROM employees 
INNER JOIN departments ON employees.department_id = departments.department_id""",
            
            "Example 2: ALL employees (shows NULLs)": """SELECT employees.name, departments.department_name, employees.salary 
FROM employees 
LEFT JOIN departments ON employees.department_id = departments.department_id""",
            
            "Example 3: High earners in Sales": """SELECT employees.name, departments.department_name, employees.salary 
FROM employees 
INNER JOIN departments ON employees.department_id = departments.department_id 
WHERE departments.department_name = 'Sales' AND employees.salary > 50000""",
            
            "Example 4: Average salary by department": """SELECT departments.department_name, AVG(employees.salary) as avg_salary 
FROM employees 
INNER JOIN departments ON employees.department_id = departments.department_id 
GROUP BY departments.department_name 
ORDER BY avg_salary DESC""",
            
            "Example 5: Department headcount": """SELECT departments.department_name, COUNT(employees.employee_id) as employee_count 
FROM departments 
LEFT JOIN employees ON departments.department_id = employees.department_id 
GROUP BY departments.department_name"""
        }
    },

    "Dataset 4: Search Trends & Categories (Time-Series)": {
    "table1_name": "search_trends",
    "table2_name": "term_categories",
    "table1_data": load_search_trends_data,
    "table2_data": load_term_categories_data,
    "join_key": "search_term",
    "description": "Master GROUP BY with time-series data - learn aggregation levels",
    "examples": {
        "Example 1: Raw detail data (NO grouping)": """SELECT search_date, search_term, region_name, search_interest 
FROM search_trends 
WHERE search_term = 'Python tutorial' 
ORDER BY search_date, region_name 
LIMIT 20""",
        
        "Example 2: Daily totals by term (1st aggregation)": """SELECT search_date, search_term, SUM(search_interest) as total_interest 
FROM search_trends 
GROUP BY search_date, search_term 
ORDER BY search_date, total_interest DESC""",
        
        "Example 3: Country-level aggregation": """SELECT country_name, search_term, AVG(search_interest) as avg_interest 
FROM search_trends 
GROUP BY country_name, search_term 
ORDER BY avg_interest DESC""",
        
        "Example 4: Overall popularity by term": """SELECT search_term, 
       COUNT(*) as data_points,
       AVG(search_interest) as avg_interest,
       MAX(search_interest) as peak_interest 
FROM search_trends 
GROUP BY search_term 
ORDER BY avg_interest DESC""",
        
        "Example 5: JOIN with aggregation": """SELECT term_categories.category, 
       COUNT(DISTINCT search_trends.search_term) as num_terms,
       AVG(search_trends.search_interest) as avg_interest 
FROM search_trends 
INNER JOIN term_categories ON search_trends.search_term = term_categories.search_term 
GROUP BY term_categories.category 
ORDER BY avg_interest DESC""",
        
        "Example 6: Multi-level aggregation (CAREFUL!)": """SELECT country_code, 
       search_term,
       COUNT(*) as row_count,
       SUM(search_interest) as total_interest 
FROM search_trends 
GROUP BY country_code, search_term 
ORDER BY total_interest DESC""",
        
        "Example 7: Time-series trend (per country)": """SELECT country_name, 
       search_date,
       AVG(search_interest) as daily_avg 
FROM search_trends 
WHERE search_term = 'Machine learning' 
GROUP BY country_name, search_date 
ORDER BY country_name, search_date""",
        
        "Example 8: Category performance with JOIN": """SELECT term_categories.category,
       term_categories.difficulty,
       search_trends.country_name,
       AVG(search_trends.search_interest) as avg_interest 
FROM term_categories 
INNER JOIN search_trends ON term_categories.search_term = search_trends.search_term 
GROUP BY term_categories.category, term_categories.difficulty, search_trends.country_name 
ORDER BY avg_interest DESC 
LIMIT 15"""
        }
    },

    "Dataset 5: Sales & Analytics (CTE Learning)": {
        "table1_name": "sales",
        "table2_name": "products",
        "table1_data": load_sales_data,
        "table2_data": load_products_data,
        "join_key": "product_id",
        "description": "Learn CTEs (WITH clause) step-by-step - build complex queries from simple parts",
        "examples": {
            "Example 1: Basic CTE - Category Totals": """WITH category_totals AS (
    SELECT category, SUM(revenue) as total_revenue
    FROM sales
    INNER JOIN products ON sales.product_id = products.product_id
    GROUP BY category
)
SELECT * FROM category_totals
ORDER BY total_revenue DESC""",

            "Example 2: CTE with Filter - Top Products": """WITH product_sales AS (
    SELECT products.product_name, SUM(sales.quantity) as total_qty, SUM(sales.revenue) as total_revenue
    FROM sales
    INNER JOIN products ON sales.product_id = products.product_id
    GROUP BY products.product_name
)
SELECT * FROM product_sales
WHERE total_revenue > 200
ORDER BY total_revenue DESC""",

            "Example 3: CTE for Regional Analysis": """WITH regional_summary AS (
    SELECT region, COUNT(*) as num_sales, SUM(revenue) as region_revenue
    FROM sales
    GROUP BY region
)
SELECT region, num_sales, region_revenue
FROM regional_summary
ORDER BY region_revenue DESC""",

            "Example 4: CTE with JOIN to Original Table": """WITH high_value_products AS (
    SELECT product_id, SUM(revenue) as total_revenue
    FROM sales
    GROUP BY product_id
    HAVING SUM(revenue) > 300
)
SELECT products.product_name, products.category, high_value_products.total_revenue
FROM high_value_products
INNER JOIN products ON high_value_products.product_id = products.product_id
ORDER BY total_revenue DESC""",

            "Example 5: Daily Trends with CTE": """WITH daily_sales AS (
    SELECT sale_date, SUM(quantity) as daily_qty, SUM(revenue) as daily_revenue
    FROM sales
    GROUP BY sale_date
)
SELECT sale_date, daily_qty, daily_revenue
FROM daily_sales
ORDER BY sale_date""",

            "Example 6: Category Performance CTE": """WITH category_metrics AS (
    SELECT products.category, 
           COUNT(*) as num_transactions,
           SUM(sales.quantity) as total_units,
           SUM(sales.revenue) as total_revenue
    FROM sales
    INNER JOIN products ON sales.product_id = products.product_id
    GROUP BY products.category
)
SELECT category, num_transactions, total_units, total_revenue
FROM category_metrics
ORDER BY total_revenue DESC"""
        }
    }
} 

# Parse SQL query into steps
def parse_sql_steps(query):
    """Extract SQL clauses in EXECUTION ORDER"""
    query_upper = query.strip().upper()
    steps = []
    
    # EXECUTION ORDER: CTE → JOIN → WHERE → SELECT → ORDER BY
    
    # For analyzing subsequent steps, use main_query (excludes CTE definition)
    main_query = query
    
    # 0. Check for WITH clause (CTE - executes first)
    if 'WITH' in query_upper:
        # Match CTE pattern: WITH name AS (query)
        cte_match = re.search(r'WITH\s+(\w+)\s+AS\s*\((.*?)\)\s*(SELECT.*)', query, re.IGNORECASE | re.DOTALL)
        if cte_match:
            steps.append({
                'type': 'CTE',
                'cte_name': cte_match.group(1).strip(),
                'cte_query': cte_match.group(2).strip(),
                'description': f"Create temporary result set '{cte_match.group(1).strip()}'"
            })
            # Use only the main query (after CTE) for subsequent parsing
            main_query = cte_match.group(3).strip()
    
    main_query_upper = main_query.upper()
    
    # 1. Check for JOIN in main query (not inside CTE)
    if 'JOIN' in main_query_upper:
        join_match = re.search(r'(LEFT JOIN|RIGHT JOIN|INNER JOIN|JOIN)\s+(\w+)\s+ON\s+(.+?)(?:WHERE|ORDER BY|GROUP BY|LIMIT|$)', 
                               main_query, re.IGNORECASE | re.DOTALL)
        if join_match:
            steps.append({
                'type': 'JOIN',
                'join_type': join_match.group(1).strip(),
                'table': join_match.group(2).strip(),
                'condition': join_match.group(3).strip(),
                'description': f'{join_match.group(1)} tables'
            })
    
    # 2. Check for WHERE clause in main query
    if 'WHERE' in main_query_upper:
        where_match = re.search(r'WHERE\s+(.+?)(?:ORDER BY|GROUP BY|LIMIT|$)', main_query, re.IGNORECASE | re.DOTALL)
        if where_match:
            steps.append({
                'type': 'WHERE',
                'clause': where_match.group(1).strip(),
                'description': 'Filter rows based on condition'
            })
    
    # 3. Check for SELECT clause in main query
    select_match = re.search(r'SELECT\s+(.+?)\s+FROM', main_query, re.IGNORECASE | re.DOTALL)
    if select_match:
        select_cols = select_match.group(1).strip()
        if select_cols != '*' and 'COUNT' not in select_cols.upper() and 'SUM' not in select_cols.upper():
            steps.append({
                'type': 'SELECT',
                'clause': select_cols,
                'description': 'Select specific columns'
            })
    
    # 4. Check for GROUP BY clause in main query
    if 'GROUP BY' in main_query_upper:
        group_match = re.search(r'GROUP BY\s+(.+?)(?:HAVING|ORDER BY|LIMIT|$)', main_query, re.IGNORECASE | re.DOTALL)
        if group_match:
            select_match = re.search(r'SELECT\s+(.+?)\s+FROM', main_query, re.IGNORECASE | re.DOTALL)
            steps.append({
                'type': 'GROUP BY',
                'clause': group_match.group(1).strip(),
                'select_clause': select_match.group(1).strip() if select_match else '*',
                'description': 'Group rows and aggregate'
            })
    
    # 5. Check for ORDER BY clause in main query
    if 'ORDER BY' in main_query_upper:
        order_match = re.search(r'ORDER BY\s+(.+?)(?:LIMIT|$)', main_query, re.IGNORECASE | re.DOTALL)
        if order_match:
            steps.append({
                'type': 'ORDER BY',
                'clause': order_match.group(1).strip(),
                'description': 'Sort results'
            })
    
    return steps

# Apply WHERE clause and return filtered dataframe
def apply_where_clause(df, where_clause):
    """Apply WHERE condition and return DataFrame with filtering info"""
    try:
        # Clean the WHERE clause - remove table prefixes since we're working with a single dataframe
        cleaned_clause = where_clause
        
        # Replace table.column with just column (e.g., "orders.amount" -> "amount")
        # Match word.word pattern
        cleaned_clause = re.sub(r'(\w+)\.(\w+)', r'\2', cleaned_clause)
        
        conn = sqlite3.connect(':memory:')
        df.to_sql('temp_table', conn, index=False, if_exists='replace')
        query = f"SELECT * FROM temp_table WHERE {cleaned_clause}"
        result = pd.read_sql_query(query, conn)
        conn.close()
        return result, None
    except Exception as e:
        return None, str(e)

# Fix duplicate column names after JOIN
def fix_duplicate_columns(df):
    """Rename duplicate columns by adding _1, _2 suffix"""
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        dup_indices = cols[cols == dup].index.tolist()
        # Rename duplicates: first stays same, others get _1, _2, etc.
        for i, idx in enumerate(dup_indices[1:], 1):
            cols[idx] = f"{dup}_{i}"
    df.columns = cols
    return df

# Main UI
df_customers = load_customers_data()
df_orders = load_orders_data()

# Sidebar for dataset and table selection
with st.sidebar:
    st.header("📚 Choose Dataset")
    
    selected_dataset_name = st.selectbox(
        "Select learning dataset:",
        list(DATASETS.keys()),
        help="Choose a dataset to practice SQL"
    )
    
    dataset_config = DATASETS[selected_dataset_name]
    
    st.info(f"ℹ️ {dataset_config['description']}")
    
    st.divider()
    
    st.header("📋 Available Tables")
    
    show_table1 = st.checkbox(f"Show {dataset_config['table1_name'].title()} Table", value=True)
    show_table2 = st.checkbox(f"Show {dataset_config['table2_name'].title()} Table", value=True)
    
    st.divider()
    
    st.markdown("### 🔗 Table Relationships")
    st.info(f"**{dataset_config['table1_name']}.{dataset_config['join_key']}** ↔️ **{dataset_config['table2_name']}.{dataset_config['join_key']}**")
    
    # Load the actual data
    df_table1 = dataset_config['table1_data']()
    df_table2 = dataset_config['table2_data']()
    
    st.divider()
    
    st.markdown("### 📊 Table Stats")
    st.metric(dataset_config['table1_name'].title(), len(df_table1))
    st.metric(dataset_config['table2_name'].title(), len(df_table2))

# Show tables at the top
if show_table1 or show_table2:
    st.subheader("📊 Original Tables")
    
    table_cols = st.columns(2)
    
    with table_cols[0]:
        if show_table1:
            st.markdown(f"**`{dataset_config['table1_name']}` table:**")
            st.dataframe(df_table1, use_container_width=True, hide_index=True)
    
    with table_cols[1]:
        if show_table2:
            st.markdown(f"**`{dataset_config['table2_name']}` table:**")
            st.dataframe(df_table2, use_container_width=True, hide_index=True)
    
    st.divider()

# SQL Query input
st.subheader("🔍 SQL Query Editor")

# Predefined query examples
# Get examples from selected dataset
example_queries = dataset_config['examples'].copy()
example_queries["Custom Query"] = ""  # Add custom query option

# Two columns for example selector and editor
editor_col1, editor_col2 = st.columns([1, 3])

with editor_col1:
    selected_example = st.selectbox("Quick Examples:", list(example_queries.keys()), key="example_selector")
    
    if st.button("📋 Load Example", use_container_width=True):
        st.session_state.query_content = example_queries[selected_example]
        st.rerun()

with editor_col2:
    st.markdown("*Write or edit your SQL query below:*")

# Initialize query content in session state
if 'query_content' not in st.session_state:
    st.session_state.query_content = example_queries["Example 1: INNER JOIN"]

# Check if user selected "Custom Query"
is_custom_query = (selected_example == "Custom Query")

if is_custom_query:
    # CUSTOM QUERY: Editable text area with basic highlighting
    st.markdown("*Write your custom SQL query:*")
    query = st.text_area(
        "",
        value=st.session_state.query_content,
        height=200,
        help="Write your SQL query here",
        placeholder="SELECT customers.name FROM customers WHERE age > 25",
        key="sql_editor",
        label_visibility="collapsed"
    )
    st.session_state.query_content = query
else:
    # EXAMPLE QUERY: Beautiful read-only display with full syntax highlighting
    st.markdown("*Example query (read-only):*")
    st.code(example_queries[selected_example], language='sql', line_numbers=True)
    query = example_queries[selected_example]
    st.session_state.query_content = query
    
    # Add a hint
    st.info("💡 **Want to modify this example?** Switch to 'Custom Query' to edit freely.")

# Buttons
col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 1])

with col_btn1:
    execute_full = st.button("▶️ Execute Full Query", type="primary", use_container_width=True)

with col_btn2:
    execute_steps = st.button("🔄 Execute Step-by-Step", type="secondary", use_container_width=True)

with col_btn3:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.query_content = ""
        st.rerun()

st.divider()

# Execute query - FULL EXECUTION
if execute_full and query.strip():
    try:
        conn = sqlite3.connect(':memory:')
        df_table1.to_sql(dataset_config['table1_name'], conn, index=False, if_exists='replace')
        df_table2.to_sql(dataset_config['table2_name'], conn, index=False, if_exists='replace')
        
        result = pd.read_sql_query(query, conn)
        
        # Fix duplicate columns if they exist
        result = fix_duplicate_columns(result)
        
        conn.close()
        
        st.subheader("✅ Query Result")
        
        result_col1, result_col2 = st.columns([2, 1])
        
        with result_col1:
            st.dataframe(result, use_container_width=True, hide_index=True)
        
        with result_col2:
            st.markdown("### 📈 Summary")
            st.metric("Rows Returned", len(result))
            st.metric("Columns Returned", len(result.columns))
        
    except Exception as e:
        st.error(f"❌ SQL Error: {str(e)}")
        with st.expander("💡 Common fixes"):
            st.markdown("""
            - Check table names: `customers`, `orders`
            - Use `table.column` notation for JOINs
            - Make sure JOIN condition matches: `customers.customer_id = orders.customer_id`
            - Check for typos in column names
            """)

# Execute query - STEP-BY-STEP EXECUTION
elif execute_steps and query.strip():
    try:
        conn = sqlite3.connect(':memory:')
        df_table1.to_sql(dataset_config['table1_name'], conn, index=False, if_exists='replace')
        df_table2.to_sql(dataset_config['table2_name'], conn, index=False, if_exists='replace')
        steps = parse_sql_steps(query)
        
        if not steps:
            st.info("ℹ️ This query returns data directly. Try adding JOIN, WHERE, or ORDER BY for step-by-step visualization.")
            
            # Still execute and show result
            result = pd.read_sql_query(query, conn)
            result = fix_duplicate_columns(result)
            conn.close()
            
            st.dataframe(result, use_container_width=True, hide_index=True)
        else:
            st.subheader("🔄 Step-by-Step Execution")
            
            # Setup database
            conn = sqlite3.connect(':memory:')
            df_table1.to_sql(dataset_config['table1_name'], conn, index=False, if_exists='replace')
            df_table2.to_sql(dataset_config['table2_name'], conn, index=False, if_exists='replace')
            
            current_df = None
            step_number = 0
            
            # NEW: Check if query has JOIN - if not, initialize with base table
            has_join = any(step['type'] == 'JOIN' for step in steps)
            
            if has_join:
                # Show original tables if there's a JOIN
                st.markdown("### 📊 Starting Tables")
                join_col1, join_col2 = st.columns(2)
                
                with join_col1:
                    st.markdown(f"**📋 {dataset_config['table1_name']} table**")
                    st.dataframe(df_table1, use_container_width=True, hide_index=True)
                
                with join_col2:
                    st.markdown(f"**📋 {dataset_config['table2_name']} table**")
                    st.dataframe(df_table2, use_container_width=True, hide_index=True)
                
                st.divider()
            else:
                # NEW: For queries without JOIN, initialize current_df with the FROM table
                # Detect which table is being queried
                from_match = re.search(r'FROM\s+(\w+)', query, re.IGNORECASE)
                if from_match:
                    from_table = from_match.group(1).strip()
                    
                    # Load the appropriate table into current_df
                    if from_table.lower() == dataset_config['table1_name'].lower():
                        current_df = df_table1.copy()
                        display_table_name = dataset_config['table1_name']
                    elif from_table.lower() == dataset_config['table2_name'].lower():
                        current_df = df_table2.copy()
                        display_table_name = dataset_config['table2_name']
                    else:
                        st.error(f"Unknown table: {from_table}")
                        current_df = df_table1.copy()
                        display_table_name = dataset_config['table1_name']
                    
                    # Show starting table
                    st.markdown("### 📊 Starting Table")
                    st.markdown(f"**📋 {display_table_name} table** ({len(current_df)} rows)")
                    st.dataframe(current_df.head(10), use_container_width=True, hide_index=True)
                    
                    if len(current_df) > 10:
                        with st.expander(f"View all {len(current_df)} rows"):
                            st.dataframe(current_df, use_container_width=True, hide_index=True)
                    
                    st.divider()
            
            for i, step in enumerate(steps):
                step_number += 1
                st.markdown(f"### Step {step_number}: {step['type']} - *{step['description']}*")
                
                if step['type'] == 'CTE':
                    st.code(f"WITH {step['cte_name']} AS (\n{step['cte_query']}\n)", language='sql')
                    
                    # Show original tables being used in CTE
                    st.markdown("**📊 Original tables used in CTE:**")
                    cte_col1, cte_col2 = st.columns(2)
                    with cte_col1:
                        st.markdown(f"**`{dataset_config['table1_name']}`**")
                        st.dataframe(df_table1.head(8), use_container_width=True, hide_index=True)
                    with cte_col2:
                        st.markdown(f"**`{dataset_config['table2_name']}`**")
                        st.dataframe(df_table2.head(8), use_container_width=True, hide_index=True)
                    
                    # Execute the CTE query to show intermediate result
                    try:
                        cte_result = pd.read_sql_query(step['cte_query'], conn)
                        cte_result = fix_duplicate_columns(cte_result)
                        
                        st.markdown(f"**🔄 CTE Result - `{step['cte_name']}`:** (This temporary result will be used in the main query)")
                        st.dataframe(cte_result, use_container_width=True, hide_index=True)
                        
                        # Register CTE result as a table for subsequent queries
                        cte_result.to_sql(step['cte_name'], conn, index=False, if_exists='replace')
                        current_df = cte_result
                        
                        st.info(f"💡 **CTE Explanation:** The WITH clause creates a temporary named result set `{step['cte_name']}` with {len(cte_result)} rows. This makes the main query simpler and more readable.")
                    except Exception as e:
                        st.error(f"Error executing CTE: {str(e)}")
                
                elif step['type'] == 'JOIN':
                    st.code(f"{step['join_type']} {step['table']} ON {step['condition']}", language='sql')
    
                    # Check if we have a CTE result (current_df is already set from CTE step)
                    has_cte = any(s['type'] == 'CTE' for s in steps)
                    
                    # Extract the actual FROM table from the main query
                    # For CTE queries, we need to look at the main query part
                    if has_cte:
                        # Get the main query (after CTE)
                        cte_match = re.search(r'WITH\s+\w+\s+AS\s*\(.*?\)\s*(SELECT.*)', query, re.IGNORECASE | re.DOTALL)
                        if cte_match:
                            main_query_part = cte_match.group(1)
                            from_match = re.search(r'FROM\s+(\w+)', main_query_part, re.IGNORECASE)
                        else:
                            from_match = re.search(r'FROM\s+(\w+)', query, re.IGNORECASE)
                    else:
                        from_match = re.search(r'FROM\s+(\w+)', query, re.IGNORECASE)
                    
                    if from_match:
                        base_table_name = from_match.group(1).strip()
                    else:
                        base_table_name = dataset_config['table1_name']  # fallback
    
                    # Check if base table is a CTE result (use current_df) or an original table
                    cte_names = [s['cte_name'].lower() for s in steps if s['type'] == 'CTE']
                    
                    if base_table_name.lower() in cte_names and current_df is not None:
                        # Base is CTE result - use current_df
                        base_df = current_df
                        base_display_name = base_table_name
                    elif base_table_name.lower() == dataset_config['table1_name'].lower():
                        base_df = df_table1
                        base_display_name = dataset_config['table1_name']
                    else:
                        base_df = df_table2
                        base_display_name = dataset_config['table2_name']
    
                    if step['table'].lower() == dataset_config['table1_name'].lower():
                        join_df = df_table1
                        join_display_name = dataset_config['table1_name']
                    elif step['table'].lower() == dataset_config['table2_name'].lower():
                        join_df = df_table2
                        join_display_name = dataset_config['table2_name']
                    else:
                        # Could be joining to a CTE
                        join_df = current_df if current_df is not None else df_table2
                        join_display_name = step['table']
    
                    # Show the two tables being joined
                    st.markdown("**Tables being joined:**")
    
                    join_col1, join_col2, join_col3 = st.columns([2, 1, 2])
    
                    with join_col1:
                        st.markdown(f"**📋 {base_display_name}**")
                        st.dataframe(base_df.head(10) if len(base_df) > 10 else base_df, use_container_width=True, hide_index=True)
    
                    with join_col2:
                        st.markdown("")
                        st.markdown("")
                        st.markdown("### 🔗")
                        st.markdown(f"**{step['join_type']}**")
                        st.markdown(f"ON `{step['condition']}`")
    
                    with join_col3:
                        st.markdown(f"**📋 {join_display_name}**")
                        st.dataframe(join_df.head(10) if len(join_df) > 10 else join_df, use_container_width=True, hide_index=True)
    
                    # Execute the JOIN - use the already-registered CTE table if applicable
                    join_query = f"SELECT * FROM {base_table_name} {step['join_type']} {step['table']} ON {step['condition']}"
                    current_df = pd.read_sql_query(join_query, conn)
    
                    # Fix duplicate columns
                    current_df = fix_duplicate_columns(current_df)
    
                    st.markdown("**After joining:**")
                    st.dataframe(current_df, use_container_width=True, hide_index=True)
    
                    # Show statistics
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric(base_display_name.title(), len(base_df))
                    with col_stat2:
                        st.metric(join_display_name.title(), len(join_df))
                    with col_stat3:
                        st.metric("Joined Rows", len(current_df))
    
                    if 'LEFT JOIN' in step['join_type']:
                        st.info(f"💡 **LEFT JOIN** keeps all {len(base_df)} {base_display_name}, even those without matches (shown as NULL)")
                    elif 'INNER JOIN' in step['join_type']:
                        st.info(f"💡 **INNER JOIN** only keeps rows with matches ({len(current_df)} rows)")

                elif step['type'] == 'WHERE':
                    st.code(step['clause'], language='sql')
                    
                    if current_df is not None:
                        st.markdown(f"**Before filtering:** {len(current_df)} rows")
                        
                        # Show sample of data before filtering
                        with st.expander("Show data before filtering"):
                            st.dataframe(current_df, use_container_width=True, hide_index=True)
                        
                        # Apply WHERE to current result
                        filtered_df, error = apply_where_clause(current_df, step['clause'])
                        
                        if filtered_df is not None:
                            # Highlight matching rows - but limit display to reasonable size
                            if len(current_df) <= 100:
                                # Reset indices to ensure proper matching
                                current_df_reset = current_df.reset_index(drop=True)
                                filtered_df_reset = filtered_df.reset_index(drop=True)
                                
                                # Create a set of tuples representing filtered rows for fast lookup
                                filtered_rows_set = set(filtered_df_reset.apply(tuple, axis=1))
                                
                                # Check which rows match
                                display_df = current_df_reset.copy()
                                display_df['✓ Matches'] = [
                                    '✅ KEEP' if tuple(row) in filtered_rows_set else '❌ REMOVE'
                                    for _, row in current_df_reset.iterrows()
                                ]
                                
                                def highlight_where(row):
                                    row_tuple = tuple(row.iloc[:-1])  # Exclude the '✓ Matches' column
                                    if row_tuple in filtered_rows_set:
                                        return ['background-color: #d4edda'] * len(row)
                                    else:
                                        return ['background-color: #f8d7da'] * len(row)
                                
                                styled_df = display_df.style.apply(highlight_where, axis=1)
                                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                            else:
                                # For large datasets, just show the filtered result
                                st.markdown(f"**Filtered data preview:**")
                                st.dataframe(filtered_df.head(20), use_container_width=True, hide_index=True)
                            
                            st.markdown(f"**After filtering:** {len(filtered_df)} rows kept, {len(current_df) - len(filtered_df)} rows removed")
                            
                            current_df = filtered_df
                        else:
                            st.error(f"Error applying WHERE: {error}")
                    else:
                        st.error("⚠️ No data available. WHERE clause needs data from a previous step.")
                
                elif step['type'] == 'SELECT':
                    # Skip SELECT step if there's a GROUP BY (handled together)
                    has_group_by = any(s['type'] == 'GROUP BY' for s in steps)
                    if has_group_by:
                        st.info("ℹ️ Column selection handled in GROUP BY step")
                        continue

                    st.code(step['clause'], language='sql')
                    
                    if current_df is not None:
                        st.markdown("**Before selecting columns:**")
                        st.dataframe(current_df.head(3), use_container_width=True, hide_index=True)
                        
                        # Parse column names - handle table prefixes
                        selected_cols_raw = [col.strip() for col in step['clause'].split(',')]
                        selected_cols = []
                        
                        for col in selected_cols_raw:
                            # Remove table prefix (e.g., "customers.name" -> "name")
                            if '.' in col:
                                col = col.split('.')[-1]
                            selected_cols.append(col)
                        
                        # Filter to columns that exist
                        selected_cols = [col for col in selected_cols if col in current_df.columns]
                        
                        if selected_cols:
                            removed_cols = [col for col in current_df.columns if col not in selected_cols]
                            current_df = current_df[selected_cols]
                            st.markdown(f"**After selecting columns:** Kept {len(selected_cols)} columns")
                            if removed_cols:
                                st.caption(f"Removed: {', '.join(removed_cols)}")
                            
                            st.markdown("**Result:**")
                            st.dataframe(current_df, use_container_width=True, hide_index=True)
                    else:
                        st.error("⚠️ No data available. SELECT needs data from a previous step.")
                
                elif step['type'] == 'GROUP BY':
                    st.code(f"GROUP BY {step['clause']}", language='sql')
                    st.code(f"SELECT {step['select_clause']}", language='sql')
                    
                    if current_df is not None:
                        rows_before = len(current_df)
                        st.markdown(f"**Before grouping:** {rows_before} rows")
                        st.dataframe(current_df.head(10), use_container_width=True, hide_index=True)
                        
                        if len(current_df) > 10:
                            with st.expander(f"View all {len(current_df)} rows before grouping"):
                                st.dataframe(current_df, use_container_width=True, hide_index=True)
                        
                        # Execute the full query with GROUP BY using SQL
                        conn_temp = sqlite3.connect(':memory:')
                        current_df.to_sql('temp_grouped', conn_temp, index=False, if_exists='replace')
                        
                        # Clean column names (remove table prefixes from SELECT and GROUP BY)
                        select_cleaned = re.sub(r'(\w+)\.(\w+)', r'\2', step['select_clause'])
                        group_cleaned = re.sub(r'(\w+)\.(\w+)', r'\2', step['clause'])
                        
                        group_query = f"SELECT {select_cleaned} FROM temp_grouped GROUP BY {group_cleaned}"
                        
                        try:
                            current_df = pd.read_sql_query(group_query, conn_temp)
                            conn_temp.close()
                            
                            st.markdown("**After grouping and aggregating:**")
                            st.dataframe(current_df, use_container_width=True, hide_index=True)
                            
                            st.info(f"💡 **GROUP BY** collapsed {rows_before} rows into {len(current_df)} groups with aggregated values")
                        except Exception as e:
                            st.error(f"Error in GROUP BY: {str(e)}")
                            conn_temp.close()
                    else:
                        st.error("⚠️ No data available. GROUP BY needs data from a previous step.")

                elif step['type'] == 'ORDER BY':
                    st.code(step['clause'], language='sql')
    
                    if current_df is not None:
                        st.markdown("**Before sorting:**")
                        st.dataframe(current_df.head(5), use_container_width=True, hide_index=True)
        
                        # Parse ORDER BY - handle multiple columns
                        order_clause = step['clause'].strip()
        
                        # Split by comma to handle multiple columns
                        order_parts = [part.strip() for part in order_clause.split(',')]
        
                        # Parse each column and its direction
                        order_columns = []
                        order_ascending = []
        
                        for part in order_parts:
                            # Check for DESC/ASC
                            is_desc = 'DESC' in part.upper()
                            is_asc = 'ASC' in part.upper()
            
                            # Remove DESC/ASC and clean
                            col = part.replace('DESC', '').replace('desc', '').replace('ASC', '').replace('asc', '').strip()
            
                            # Remove table prefix if present
                            if '.' in col:
                                col = col.split('.')[-1]
            
                            if col and col in current_df.columns:
                                order_columns.append(col)
                                order_ascending.append(not is_desc)  # If DESC, ascending=False; otherwise True
                            elif col:
                                st.warning(f"⚠️ Column '{col}' not found in data")
        
                        if order_columns:
                            # Sort by all columns
                            current_df = current_df.sort_values(
                                by=order_columns, 
                                ascending=order_ascending
                            ).reset_index(drop=True)
            
                            # Create readable description
                            if len(order_columns) == 1:
                                direction = 'ascending ⬆️' if order_ascending[0] else 'descending ⬇️'
                                st.markdown(f"**After sorting:** Ordered by `{order_columns[0]}` {direction}")
                            else:
                                order_desc = []
                                for col, asc in zip(order_columns, order_ascending):
                                    direction = '⬆️' if asc else '⬇️'
                                    order_desc.append(f"`{col}` {direction}")
                                st.markdown(f"**After sorting:** Ordered by {', then '.join(order_desc)}")

                            st.markdown("**Result:**")
                            st.dataframe(current_df, use_container_width=True, hide_index=True)
                        else:
                            st.error(f"No valid columns found in ORDER BY clause")
                    else:
                        st.error("⚠️ No data available. ORDER BY needs data from a previous step.")
          
                st.divider()
            
            conn.close()
            
            # Final result
            if current_df is not None:
                st.subheader("✅ Final Result")
                st.dataframe(current_df, use_container_width=True, hide_index=True)
                st.success(f"**Final output:** {len(current_df)} rows × {len(current_df.columns)} columns")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        with st.expander("Show error details"):
            st.code(traceback.format_exc())

elif (execute_full or execute_steps) and not query.strip():
    st.warning("⚠️ Please enter a SQL query first!")

# Footer with tips (make it dynamic based on dataset)
st.divider()
st.markdown("### 💡 Quick Reference")

if selected_dataset_name == "Dataset 4: Search Trends & Categories (Time-Series)":
    tip_col1, tip_col2, tip_col3, tip_col4 = st.columns(4)
    
    with tip_col1:
        st.markdown("""
        **search_trends columns:**
        - `search_date`
        - `search_term`
        - `country_code/name`
        - `region_code/name`
        - `search_interest`
        """)
    
    with tip_col2:
        st.markdown("""
        **term_categories columns:**
        - `search_term`
        - `category`
        - `difficulty`
        """)
    
    with tip_col3:
        st.markdown("""
        **Key GROUP BY concepts:**
        - **Detail level**: 490 rows (date × term × region)
        - **Aggregate up**: Use SUM/AVG
        - **Granularity**: Daily→Weekly, Region→Country
        - **Multi-dimension**: Time + Geography + Term
        """)
    
    with tip_col4:
        st.markdown("""
        **Common mistakes:**
        - Forgetting to GROUP BY all dimensions
        - Using wrong aggregation (SUM vs AVG)
        - Double-counting in JOINs
        - Not understanding data granularity
        """)
elif selected_dataset_name == "Dataset 5: Sales & Analytics (CTE Learning)":
    tip_col1, tip_col2, tip_col3, tip_col4 = st.columns(4)
    
    with tip_col1:
        st.markdown("""
        **sales columns:**
        - `sale_id`
        - `product_id`
        - `region`
        - `sale_date`
        - `quantity`
        - `revenue`
        """)
    
    with tip_col2:
        st.markdown("""
        **products columns:**
        - `product_id`
        - `product_name`
        - `category`
        - `unit_price`
        """)
    
    with tip_col3:
        st.markdown("""
        **CTE Syntax:**
        ```sql
        WITH cte_name AS (
            SELECT ...
        )
        SELECT * FROM cte_name
        ```
        """)
    
    with tip_col4:
        st.markdown("""
        **CTE Benefits:**
        - Breaks complex queries into steps
        - Improves readability
        - Can be referenced multiple times
        - Easier to debug and test
        """)
else:
    # Original footer for other datasets
    tip_col1, tip_col2, tip_col3, tip_col4 = st.columns(4)
    
    with tip_col1:
        st.markdown("""
        **Customers columns:**
        - `customer_id`
        - `name`
        - `country`
        - `age`
        """)
    
    with tip_col2:
        st.markdown("""
        **Orders columns:**
        - `order_id`
        - `customer_id`
        - `product`
        - `amount`
        """)
    
    with tip_col3:
        st.markdown("""
        **JOIN types:**
        - `INNER JOIN` - Only matches
        - `LEFT JOIN` - All from left table
        - Use: `table.column` notation
        """)
    
    with tip_col4:
        st.markdown("""
        **Keyboard shortcuts:**
        - `Ctrl+Enter` - Run query
        - `Tab` - Indent
        - `Shift+Tab` - Unindent
        """)
