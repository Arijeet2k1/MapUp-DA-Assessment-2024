import pandas as pd
data = pd.read_csv("../datasets/dataset-2.csv")
import datetime

# Question no 9
def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Check the columns in the DataFrame
    print("Columns in DataFrame:", df.columns)
    
    # Strip whitespace from column names (if any)
    df.columns = df.columns.str.strip()

    # Get unique IDs
    unique_ids = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))
    
    # Create a distance matrix filled with infinity
    distance_matrix = pd.DataFrame(index=unique_ids, columns=unique_ids, data=float('inf'))
    
    # Fill the distance matrix with known distances
    for _, row in df.iterrows():
        start = row['id_start']
        end = row['id_end']
        distance = row['distance']
        
        # Set the distance in both directions
        distance_matrix.at[start, end] = min(distance_matrix.at[start, end], distance)
        distance_matrix.at[end, start] = min(distance_matrix.at[end, start], distance)
    
    # Apply the Floyd-Warshall algorithm to compute cumulative distances
    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                if distance_matrix.at[i, k] < float('inf') and distance_matrix.at[k, j] < float('inf'):
                    distance_matrix.at[i, j] = min(distance_matrix.at[i, j], distance_matrix.at[i, k] + distance_matrix.at[k, j])
    
    # Set diagonal values to 0
    for id in unique_ids:
        distance_matrix.at[id, id] = 0

    # Replace inf with NaN for better clarity (optional)
    distance_matrix.replace(float('inf'), pd.NA, inplace=True)

    return distance_matrix
# Input Below
Distance_Matrix_Calculation=calculate_distance_matrix(data)
print(Distance_Matrix_Calculation)

# Question no 10

def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    ids = df.index.tolist()  # IDs from the index (first column)
    distances = []
    
    # Loop through each combination of IDs
    for i in range(len(ids)):
        for j in range(len(ids)):
            if i != j:  # Exclude same id_start and id_end
                distances.append({
                    'id_start': ids[i],
                    'id_end': ids[j],
                    'distance': df.iloc[i, j]  # Use i, j directly since no offset is needed
                })
    
    # Create a new DataFrame from the list of dictionaries
    result_df = pd.DataFrame(distances)
    
    return result_df

# Input Below
unrolled_df = unroll_distance_matrix(Distance_Matrix_Calculation)
print(unrolled_df)

# Question no 11
def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Filter the DataFrame for the given reference_id
    filtered_df = df[df['id_start'] == reference_id]
    
    # Calculate the average distance for the reference_id
    if filtered_df.empty:
        return pd.DataFrame(columns=['id_start', 'average_distance'])  # Return an empty DataFrame if not found
    
    average_distance = filtered_df['distance'].mean()
    
    # Calculate the lower and upper thresholds (10% of average_distance)
    lower_threshold = average_distance * 0.9
    upper_threshold = average_distance * 1.1
    
    # Find all id_start values that have average distances within the thresholds
    ids_within_threshold = df[(df['distance'] >= lower_threshold) &
                               (df['distance'] <= upper_threshold)]['id_start']
    
    # Calculate average distance for each id_start
    average_distances = df.groupby('id_start')['distance'].mean().reset_index()
    
    # Filter the average distances to include only those within the threshold
    result_df = average_distances[average_distances['distance'].between(lower_threshold, upper_threshold)]
    
    return result_df

# Input Below
print(find_ids_within_ten_percentage_threshold(unrolled_df,1001466))

# Question no 12

def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Define rate coefficients for each vehicle type
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    # Calculate toll rates for each vehicle type
    for vehicle, rate in rates.items():
        df[vehicle] = df['distance'] * rate  # Add a new column for each vehicle type

    return df
# I haven't droped distance column as given in the readme image because I have to use it in the 13 no Question

# Input Below
toll_rates = calculate_toll_rate(unrolled_df)
print(toll_rates)

# Question no 13
def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Define days of the week
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekends = ['Saturday', 'Sunday']
    
    # Define time intervals
    time_intervals = [
        (datetime.time(0, 0), datetime.time(10, 0)),  # 00:00:00 - 10:00:00
        (datetime.time(10, 0), datetime.time(18, 0)),  # 10:00:00 - 18:00:00
        (datetime.time(18, 0), datetime.time(23, 59, 59))  # 18:00:00 - 23:59:59
    ]
    
    # Define discount factors for weekdays
    discount_factors_weekdays = [0.8, 1.2, 0.8]
    discount_factor_weekend = 0.7

    result = []

    # Loop over each row in the input DataFrame
    for idx, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']  # Add distance from the original dataframe
        
        # Loop over each day of the week
        for day in weekdays + weekends:
            # Loop over each time interval
            for i, (start_time, end_time) in enumerate(time_intervals):
                # Determine the discount factor
                if day in weekdays:
                    discount_factor = discount_factors_weekdays[i]
                else:
                    discount_factor = discount_factor_weekend
                
                # Calculate the toll rates for each vehicle type
                toll_rates = {
                    'moto': row['moto'] * discount_factor,
                    'car': row['car'] * discount_factor,
                    'rv': row['rv'] * discount_factor,
                    'bus': row['bus'] * discount_factor,
                    'truck': row['truck'] * discount_factor
                }
                
                # Append the result for this day and time interval
                result.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'distance': distance,  # Include the distance column
                    'start_day': day,
                    'start_time': start_time,
                    'end_day': day,
                    'end_time': end_time,
                    **toll_rates  # Add the calculated toll rates
                })

    # Convert the result list to a DataFrame
    time_based_toll_df = pd.DataFrame(result)

    return time_based_toll_df

# Input Below
time_based_toll_rates_df = calculate_time_based_toll_rates(toll_rates)
print(time_based_toll_rates_df)
