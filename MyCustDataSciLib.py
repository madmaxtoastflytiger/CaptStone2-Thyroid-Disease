import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2_contingency
from kmodes.kmodes import KModes



# Data Wrangling # 

# Col Names
def replace_spaces_in_col_names(df, columns='all'):

    if columns == 'all':
        columns = df.columns  # Use all columns if 'all' is specified
    
    new_columns = {col: '_'.join(col.split()) if col in columns else col for col in df.columns}
    df.rename(columns=new_columns, inplace=True)
    return df


def convert_col_names_to_lowercase(df, columns='all'):

    if columns == 'all':
        columns = df.columns  # Use all columns if 'all' is specified
    
    new_columns = {col: col.lower() if col in columns else col for col in df.columns}
    df.rename(columns=new_columns, inplace=True)
    return df


def rename_col_names(df, rename_dict):
    for old_name, new_name in rename_dict.items():

        # Check if the source column exists
        if old_name not in df.columns:
            print(f"Warning: Column '{old_name}' not found in DataFrame.")

        # Check if the target column name already exists
        if new_name in df.columns:
            print(f"Warning: Column '{new_name}' already exists in DataFrame. Conversion might already be done.")
            continue

        # Rename the column
        df.rename(columns={old_name: new_name}, inplace=True)
        print(f"Column '{old_name}' successfully renamed to '{new_name}'.")

    return df



# Col Values
def replace_spaces_in_col_values(df, columns='all'):
    if columns == 'all':
        columns = df.columns  # Use all columns if 'all' is specified
    df[columns] = df[columns].applymap(lambda x: '_'.join(x.split()) if isinstance(x, str) else x)
    return df


def convert_col_values_to_lowercase(df, columns='all'):
    if columns == 'all':
        columns = df.columns  # Use all columns if 'all' is specified
    df[columns] = df[columns].applymap(lambda x: x.lower() if isinstance(x, str) else x)
    return df


def replace_col_values(df, columns='all', replacements=None):
    '''
    replacements uses a dict
    for example: 
        {'f':'g', 2:15 }
    works with either numeric or string values, and do both at the same time
    '''
    if replacements is None:
        raise ValueError("Replacements dictionary cannot be None.")

    # Validate columns
    if columns == 'all':
        columns = df.columns  # Use all columns if 'all' is specified
    elif isinstance(columns, list):
        for col in columns:
            if col not in df.columns:
                print(f"Warning: Column '{col}' not found in DataFrame.")
    else:
        raise TypeError("Columns should be 'all' or a list of column names.")

    # Check and apply replacements
    for old_value, new_value in replacements.items():
        if old_value not in df[columns].values:
            print(f"Warning: Value '{old_value}' not found in the specified columns.")
            continue

        df[columns] = df[columns].replace({old_value: new_value})
        print(f"Replaced all occurrences of '{old_value}' with '{new_value}'.")

    return df



# Detect Outliers
def detect_outliers_in_col(df, col_name):
    
    print(f"Detecting outliers for the '{col_name}' column.")

    # Check if the column exists in the DataFrame
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found in the DataFrame.")
    
    # Calculate Q1, Q3, and IQR for the specified column
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Find outliers: values outside the bounds
    outliers = df[(df[col_name] < lower_bound) | (df[col_name] > upper_bound)]
    
    # Get the list of outlier rows and values as tuples (index, value)
    outlier_rows_info = list(outliers[[col_name]].itertuples(index=True, name=None))
    
    # List of just the outlier rows
    outlier_rows = outliers.index.tolist()

    # print out outlier analysis
    print(f"Lower bounds:   {lower_bound}")
    print(f"Q1:             {Q1}")
    print(f"Q3:             {Q3}")
    print(f"Upper bounds:   {upper_bound}")
    
    # Determine if there are any outliers and print the appropriate message
    if len(outlier_rows) == 0:
        print(f"'{col_name}' column has no outliers.")
    else:
        print(f"'{col_name}' column has {len(outlier_rows)} outliers.")
    
    # Return a tuple with the requested information
    return outlier_rows_info, outlier_rows


def test_generate_outlier_df():
    data = {
        'A': [10, 12, 14, 1000, 15, 13, 11, 18],
        'B': [1, 2, 3, 4, 5, 1000, 7, 8],
        'C': [50, 55, 60, 120, 65, 70, 75, 80]
    }

    df_outliers = pd.DataFrame(data)

    return df_outliers


# reorder columns 
def reorder_column_next_to(df, column_to_move, ref_column_move_next_to):
    if column_to_move not in df.columns or ref_column_move_next_to not in df.columns:
        raise ValueError("Both column_to_move and reference_column must exist in the DataFrame.")
    
    # Get the current column order
    columns = list(df.columns)
    
    # Remove the column to move
    columns.remove(column_to_move)
    
    # Insert it next to the reference column
    reference_index = columns.index(ref_column_move_next_to)
    columns.insert(reference_index + 1, column_to_move)
    
    # Return the reordered DataFrame
    return df[columns]





# EDA #

def visualize_outliers(df, col_name):
    plt.figure(figsize=(6, 4))  # Adjusted size for a smaller plot
    sns.boxplot(
        x=df[col_name], 
        whis=1.5, 
        showmeans=True,  # Optional to show mean as a dot
        flierprops={"marker": "o", "color": "red", "markersize": 5}  # Style for outliers
    )
    plt.title(f'Box-and-Whisker Plot for {col_name}', fontsize=14)
    plt.xlabel(col_name, fontsize=12)
    plt.show()

def quick_bar_graph(df, col_name):
    value_counts = df[col_name].value_counts()
    total_count = value_counts.sum()
    
    # Generate the title for the graph
    title = f"Bar Graph of '{col_name}' Column"
    print(f"Graph Title: {title}")  # Print the title for easy copy-paste
    
    ax = value_counts.plot(kind='bar', figsize=(10, 5))
    plt.title(title)
    plt.xlabel(col_name)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Flag for checking categories with less than 5%
    has_small_percentage = False
    
    # Annotating count and percentage on top of bars
    for idx, value in enumerate(value_counts):
        percentage = (value / total_count) * 100
        if percentage < 7.5:
            has_small_percentage = True
        ax.text(idx, value + 0.1, f"{value} ({percentage:.1f}%)", ha='center', va='bottom', fontsize=9)
    
    # Print a message if any category has less than 5%
    if has_small_percentage:
        print("Note: Some categories have less than 10% representation.")

    plt.show()
    
    


def quick_plot_all_categorical_col(df):
    # Identify categorical columns explicitly, excluding numeric types
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col])]

    if len(categorical_cols) == 0:
        print("No categorical columns found in the DataFrame.")
        return

    for col in categorical_cols:
        quick_bar_graph(df, col)


def quick_histogram(df, col_name, histo_bins=50):
    # reminder if not enough data points for bins then will have empty bars inbetween. bins merge existing data bar

    # df[col_name].plot(kind='hist', bins=histo_bins, figsize=(5, 2.5), edgecolor='black') # old code
    
    plt.figure(figsize=(5, 2.5))
    sns.histplot(df[col_name], bins=histo_bins, kde=True, edgecolor='black')
    plt.title(f"Histogram of '{col_name}' Column")
    plt.xlabel(col_name)
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()

def quick_plot_all_numeric_col(df, histo_bins=50):
    # Identify numeric columns in the DataFrame
    numeric_columns = df.select_dtypes(include=['number']).columns

    if len(numeric_columns) == 0:
        print("No numeric columns found in the DataFrame.")
        return

    # Iterate through each numeric column and plot the histogram
    for col in numeric_columns:
        print(f"Plotting for column: {col}")
        quick_histogram(df, col, histo_bins)


# Create a DataFrame with different data types in each column
def test_generate_diff_dtype_col_df():
    data = {
        'integers': [1, 2, 3, 4, 5],  # Integer column
        'floats': [1.1, 2.2, 3.3, 4.4, 5.5],  # Float column
        'strings': ['a', 'b', 'c', 'd', 'e'],  # String (object) column
        'categories': pd.Categorical(['low', 'medium', 'high', 'medium', 'low']),  # Categorical column
        'booleans': [True, False, True, False, True],  # Boolean column
        'dates': pd.to_datetime(['2022-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01']),  # Datetime column
        'timedelta': pd.to_timedelta([1, 2, 3, 4, 5], unit='D'),  # Timedelta column
        'complex_numbers': [complex(1, 1), complex(2, 2), complex(3, 3), complex(4, 4), complex(5, 5)]  # Complex numbers
    }

    df_diff_dtype_col = pd.DataFrame(data)

    return df_diff_dtype_col


def quick_stacked_bar_graph (df, col, col_in_each_stack, figsize=(5, 5)):
    # Create a contingency table
    contingency_table = pd.crosstab(df[col_in_each_stack], df[col], normalize='columns') * 100

    # Plot the stacked bar chart
    contingency_table.T.plot(kind='bar', stacked=True, figsize=figsize, colormap='viridis')

    # print title in output text to easily copy the column name
    print(f"Proportion of '{col_in_each_stack}' Categories Across '{col}'")

    # Add titles and labels
    plt.title(f"Proportion of '{col_in_each_stack}' Categories Across '{col}'", fontsize=10)
    plt.xlabel(f"'{col}' column", fontsize=10)
    plt.ylabel(f"Percentage of '{col_in_each_stack}' Categories", fontsize=10)
    plt.legend(title=col_in_each_stack, bbox_to_anchor=(1.05, 1), loc='upper left')
    #plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def quick_plot_all_stacked_bar(df, col_to_compare_to_all, figsize=(5, 5) ):

    # Identify categorical columns explicitly, excluding numeric types
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col])]

    if len(categorical_cols) == 0:
        print("No categorical columns found in the DataFrame.")
        return

    for col in categorical_cols:

        # does not plot a graph of itself vs itself
        if col == col_to_compare_to_all:
            continue # skips over
        
        # actually plotting 
        quick_stacked_bar_graph(df, col_to_compare_to_all, col, figsize)



def quick_chi_square_testing(df, col1, col2, set_p_value = 0.05): 
    # reminder - Chi-Square testing is based on comparing observed frequencies to expected frequencies under the assumption that the two variables are independent. And if the p value is lower than 5% means that the 2 columns are possibly dependent 

    print(f"chi_square_testing between '{col1}' and '{col2}'")

    # Create a contingency table
    contingency_table = pd.crosstab(df[col1], df[col2])

    # Perform chi-square test
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    # if p value is smaller than set threshold then signifigance is True
    p_significance = 'FALSE'
    if p_value < set_p_value:
        p_significance = 'TRUE'

    print("Chi-square Statistic:", chi2_stat)
    print(f"p-value: {p_value} , which is {p_significance}")
    print("Degrees of Freedom:", dof)
    print()

    if p_value < set_p_value:
        return (col1, col2)


def chi_square_test_all_col(df, col1, columns='all', pvalue=0.05):
    # If 'all', use all columns
    if columns == 'all':
        columns = df.columns  # Use all columns if 'all' is specified

    significant_columns = []  # List to store significant column pairs

    for each_col in columns:
        # Skip the comparison if col1 and each_col are the same
        if col1 == each_col:
            continue

        # Check if the column's dtype is object, category, or string
        if pd.api.types.is_string_dtype(df[each_col]) or pd.api.types.is_categorical_dtype(df[each_col]):
            # Call the chi-square testing function
            result = quick_chi_square_testing(df, col1, each_col, set_p_value=pvalue)

            # Add significant column pairs to the list
            if result is not None:
                significant_columns.append(result)

    return significant_columns
    

def cramers_v(confusion_matrix):
    # Calculate Cramér's V for each pair of categorical variables
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.values.sum()  # Ensure scalar sum
    if n == 0:
        return 0  # Avoid division by zero
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


def categorical_correlation_heatmap(df, columns='all', threshold=0.5):
    """
    Generates a correlation heatmap for categorical columns in a DataFrame and returns
    column pairs with strong Cramér's V association.

    Parameters:
    - df: pandas DataFrame containing the data.
    - columns: list of column names to include, or 'all' to use all columns.
    - threshold: threshold value for Cramér's V to consider association as strong.

    Returns:
    - Heatmap of correlation between categorical columns.
    - List of column pairs with Cramér's V > threshold.
    """
    # Select columns
    if columns == 'all':
        columns = df.columns
    else:
        columns = [col for col in columns if col in df.columns]
    
    # Filter categorical columns
    categorical_cols = [col for col in columns if df[col].dtype == 'object' or df[col].dtype.name == 'category']
    if not categorical_cols:
        raise ValueError("No categorical columns found in the specified input.")

    n = len(categorical_cols)
    correlation_matrix = np.zeros((n, n))
    strong_associations = []

    for i in range(n):
        for j in range(n):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                confusion_matrix = pd.crosstab(df[categorical_cols[i]], df[categorical_cols[j]])
                value = cramers_v(confusion_matrix)
                correlation_matrix[i, j] = value
                if value > threshold and i < j:  # Avoid duplicate pairs
                    strong_associations.append((categorical_cols[i], categorical_cols[j], value))
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", xticklabels=categorical_cols, yticklabels=categorical_cols, cmap="coolwarm")
    plt.title("Categorical Correlation Heatmap (Cramér's V)")
    plt.show()

    return strong_associations


def quick_add_kmode_cluster_col(df, n_clusters=3, init='Huang', n_init=5, verbose=1, kmode_col_name = 'kmode_cluster'):
    # Initialize the k-modes model
    kmodes = KModes(n_clusters=n_clusters, init=init, n_init=n_init, verbose=verbose)

    # Fit the model and predict cluster assignments
    clusters = kmodes.fit_predict(df)

    # Make a copy of df
    df_kmode = df.copy()

    # Add cluster assignments to the dataframe
    df_kmode['kmode_cluster'] = clusters

    # Convert the 'kmode_cluster' column to a category (or object)
    df_kmode[kmode_col_name] = df_kmode['kmode_cluster'].astype('category')  # For category type as works with my other code
    # Alternatively, for object type: df_kmode['kmode_cluster'] = df_kmode['kmode_cluster'].astype('object')

    # Print the resulting dataframe
    print(df_kmode)

    # Print cluster centroids
    print("Cluster centroids:")
    print(kmodes.cluster_centroids_)

    return df_kmode



# Feature Engineering #


# Modeling #



# Other # 

# detect unique values & if col have binary and unary data 
def print_unique_values_summary(df, selected_col='all'):
    '''
    Parameters:
        df (pd.DataFrame): Input dataframe.
        selected_col: either 'all' will do all columns, or a list of columns like ['col1','col2'], this is needed if doing one column, kept like this incase a column is
    '''

    unary_columns = []  # List to store tuples of columns with exactly one unique value

    # If 'all' is passed, use all columns in the DataFrame
    if selected_col == 'all':
        selected_col = df.columns.tolist()

    # Check if selected columns exist in the DataFrame
    for col_name in selected_col:
        if col_name in df.columns:
            # Get unique values and their counts
            unique_counts = df[col_name].value_counts()

            # Reset index to get a DataFrame and rename columns properly
            unique_count_df = unique_counts.reset_index()
            unique_count_df.columns = [col_name, 'count']

            print(f"Unique values and counts for column: '{col_name}'")
            print(unique_count_df)
            print("-" * 40)  # Separator for readability

            # Check if the column has exactly one unique value
            if len(df[col_name].unique()) == 1:
                # Get the single unique value
                unique_value = df[col_name].unique()[0]

                # Create a tuple with column name and the single unique value
                unary_columns.append((col_name, str(unique_value)))
        else:
            print(f"Column '{col_name}' does not exist in the DataFrame.")
            print("-" * 40)  # Separator for readability

    # Print warning if there are columns with only one unique value
    if unary_columns:
        print("WARNING: Detect columns with only 1 unique value.")
        print("The following displayed is a list of (col name, the one unique value)")
        print(unary_columns)
        print("-" * 40)  # Separator for readability
    else:
        print("FUNCTION FINISHED, detected no columns with only 1 unique values which is good.")


def get_binary_columns(df, selected_columns='all'):
    binary_column_info = []  # List to store tuples of columns with exactly two unique values
    binary_column_names = []  # List to store the names of binary columns

    # If 'all' is passed, use all columns in the DataFrame
    if selected_columns == 'all':
        selected_columns = df.columns.tolist()

    # Check if selected columns exist in the DataFrame
    for col_name in selected_columns:
        if col_name in df.columns:
            # Check if the column has exactly two unique values
            if len(df[col_name].unique()) == 2:
                # Extract the two unique values
                unique_values = df[col_name].unique()
                binary_value_1, binary_value_2 = unique_values[0], unique_values[1]

                # Create a tuple with column name and the two unique values
                binary_column_info.append((col_name, str(binary_value_1), str(binary_value_2)))
                # Append the column name to the binary column names list
                binary_column_names.append(col_name)

    return binary_column_info, binary_column_names


def get_unary_columns(df, selected_columns='all'):
    unary_column_info = []  # List to store tuples of columns with exactly one unique value
    unary_column_names = []  # List to store the names of unary columns

    # If 'all' is passed, use all columns in the DataFrame
    if selected_columns == 'all':
        selected_columns = df.columns.tolist()

    # Check if selected columns exist in the DataFrame
    for col_name in selected_columns:
        if col_name in df.columns:
            # Check if the column has exactly one unique value
            if len(df[col_name].unique()) == 1:
                # Get the single unique value
                unique_value = df[col_name].unique()[0]

                # Create a tuple with column name and the single unique value
                unary_column_info.append((col_name, str(unique_value)))
                # Append the column name to the unary column names list
                unary_column_names.append(col_name)

    return unary_column_info, unary_column_names


def test_generate_binary_unary_df():
    data = {
        'A': [3,4,3,4,3,4,3,4],
        'B': [7,8,7,8,7,8,7,8],
        'C': [0,0,0,0,0,0,0,0],
        'D': [1, 2, 3, 4, 5, 1000, 7, 8]
    }

    df_binary_unary = pd.DataFrame(data)

    return df_binary_unary


# Convert specified values in a column to binary 1 or 0
def merge_into_new_binary_col(df, col_name, values_to_cat1, values_to_cat0, new_col_name, category_1, category_0):
    '''
    Converts specified values in a column to user-defined categories (category_1 and category_0),
    adds the result to a new column, and returns the updated dataframe along with lists of
    incompatible values and rows.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        col_name (str): Column to modify.
        values_to_cat1 (str or list): Value(s) to be converted to category_1.
        values_to_cat0 (str or list): Value(s) to be converted to category_0.
        category_1 (str): The label for the first category.
        category_0 (str): The label for the second category.
        new_col_name (str): Name of the new column to add with the converted categories.

    Returns:
        pd.DataFrame: Updated dataframe.
        list: List of incompatible value tuples (row index, value).
        list: List of row indices with incompatible values.
    '''

    # Normalize input lists (convert to list if not already iterable)
    if isinstance(values_to_cat1, str):
        values_to_cat1 = [values_to_cat1]
    if isinstance(values_to_cat0, str):
        values_to_cat0 = [values_to_cat0]

    # Normalize the input lists (convert all elements to lowercase strings for consistency)
    normalized_values_to_cat1 = {str(val).strip().lower() for val in values_to_cat1}
    normalized_values_to_cat0 = {str(val).strip().lower() for val in values_to_cat0}

    incompatible_rows_info = []  # List to record incompatible values (as tuples)
    incompatible_rows = []  # List to record rows with incompatible values

    # Check for incompatible values
    for idx, value in df[col_name].items():
        # Normalize the value from the dataframe
        cleaned_value = str(value).strip().lower()

        # Check if the value is neither in values_to_cat1 nor values_to_cat0
        if cleaned_value not in normalized_values_to_cat1 | normalized_values_to_cat0:
            incompatible_rows_info.append((idx, value))  # Record incompatible value with row index
            incompatible_rows.append(idx)  # Record row index of incompatible value

    # If there is at least one incompatible value, return the original dataframe and lists
    if len(incompatible_rows_info) != 0:
        print(f'{col_name} column has {len(incompatible_rows)} incompatible rows')
        return df, incompatible_rows_info, incompatible_rows
    else:
        # Create a new column with the specified categories
        df[new_col_name] = df[col_name].apply(
            lambda x: category_1 if str(x).strip().lower() in normalized_values_to_cat1
            else (category_0 if str(x).strip().lower() in normalized_values_to_cat0 else x)
        )

        # Feedback that conversion was successful
        print(f'New column "{new_col_name}" has been added with the converted categories')

    # Return the updated dataframe and the lists of incompatibles
    return df, incompatible_rows_info, incompatible_rows


# Convert specified values in a column to binary 1 or 0
def convert_column_to_binary(df, col_name, values_to_1, values_to_0,):
    '''
    Converts specified values to binary 1 and 0 based on user input (as strings or lists)
    and returns the updated dataframe, along with lists of incompatible values and rows.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        col_name (str): Column to modify.
        values_to_1 (str or list): Value(s) to be converted to 1.
        values_to_0 (str or list): Value(s) to be converted to 0.

    Returns:
        pd.DataFrame: Updated dataframe.
        list: List of incompatible value tuples (row index, value).
        list: List of row indices with incompatible values.
    '''

    # Normalize input lists (convert to list if not already iterable)
    if isinstance(values_to_1, str):
        values_to_1 = [values_to_1]
    if isinstance(values_to_0, str):
        values_to_0 = [values_to_0]

    # Normalize the input lists (convert all elements to lowercase strings for consistency)
    normalized_values_to_1 = {str(val).strip().lower() for val in values_to_1}
    normalized_values_to_0 = {str(val).strip().lower() for val in values_to_0}

    incompatible_rows_info = []  # List to record incompatible values (as tuples)
    incompatible_rows = []  # List to record rows with incompatible values

    # Check if the column already contains values 0 or 1
    if df[col_name].isin([0, 1]).all():
        print(f'{col_name} column has already been converted thus no action is taken')
        return df, incompatible_rows_info, incompatible_rows

    # Iterate through the dataframe
    for idx, value in df[col_name].items():
        # Normalize the value from the dataframe
        cleaned_value = str(value).strip().lower()

        # Check if the value is neither in values_to_1 nor values_to_0, and not already binary
        if cleaned_value not in normalized_values_to_1 | normalized_values_to_0 | {'1', '0'}:
            incompatible_rows_info.append((idx, value))  # Record incompatible value with row index
            incompatible_rows.append(idx)  # Record row index of incompatible value

    # If there is at least one incompatible value, return the original dataframe and lists
    if len(incompatible_rows_info) != 0:
        print(f'{col_name} column has {len(incompatible_rows)} incompatible rows')
        return df, incompatible_rows_info, incompatible_rows
    else:
        # Otherwise, replace the specified values_to_1 with 1 and values_to_0 with 0
        df[col_name] = df[col_name].apply(
            lambda x: 1 if str(x).strip().lower() in normalized_values_to_1
            else (0 if str(x).strip().lower() in normalized_values_to_0 else x)
        )

        # Convert the column data type to int if no incompatibles
        df[col_name] = df[col_name].astype(int)

        # Feedback that conversion was successful
        print(f'{col_name} column has been fully converted')

    # Return the updated dataframe and the lists of incompatibles
    return df, incompatible_rows_info, incompatible_rows





