from typing import List, Dict
import pandas as pd
import polyline
import math
import re
df=pd.read_csv("../datasets/dataset-1.csv")


# Question 1
def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []
    for i in range(0, len(lst), n):
        group = []
        for j in range(min(n, len(lst) - i)):
            group.insert(0, lst[i + j])
        result.extend(group)
    lst = result  # Assigning the reversed result back to lst
    return lst

# Input Below
print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], n=3))

# Question 2
def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    dict = {}
    for word in lst:        
        length = len(word)
        if length not in dict:
            dict[length] = []
        dict[length].append(word)
    
    return dict  # Return the actual result dictionary

# Input Below
print(group_by_length(["apple", "bat", "car", "elephant", "dog", "beaaar"]))

# Question 3
def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    def _flatten(current, parent_key=''):
        items = {}
        for k, v in current.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, Dict):
                items.update(_flatten(v, new_key))
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, Dict):
                        items.update(_flatten(item, f"{new_key}[{i}]"))
                    else:
                        items[f"{new_key}[{i}]"] = item
            else:
                items[new_key] = v
        return items

    dict = _flatten(nested_dict)
    
    return dict

# Input Below
print(flatten_dict({
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}))

# Question 4
def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(current_permutation, remaining_nums):
        if not remaining_nums:
            result.append(current_permutation[:])
            return
        
        seen = set()
        for i in range(len(remaining_nums)):
            if remaining_nums[i] not in seen:
                seen.add(remaining_nums[i])
                current_permutation.append(remaining_nums[i])
                backtrack(current_permutation, remaining_nums[:i] + remaining_nums[i+1:])
                current_permutation.pop()

    result = []
    nums.sort()
    backtrack([], nums)

    return result

    pass  # 'pass' placed after the return statement as per the directions given in the template 1

# Input Below
print(unique_permutations([1, 1, 2]))

# Question 5
def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    # Defineed regular expression patterns for the three date formats
    date_pattern = r"\b(\d{2}-\d{2}-\d{4})\b|\b(\d{2}/\d{2}/\d{4})\b|\b(\d{4}\.\d{2}\.\d{2})\b"
    
    # Used re.findall to get all matching dates
    matches = re.findall(date_pattern, text)
    
    # Flatten the list and remove empty strings
    dates = [match for group in matches for match in group if match]
    
    return dates

    pass  # 'pass' placed after the return statement as per the directions given in the template 1

# Input Below
text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
print(find_all_dates(text))  # Output: ['23-08-1994', '08/23/1994', '1994.08.23']

# Question 6
def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    # Decode the polyline string into a list of (latitude, longitude) tuples
    coordinates = polyline.decode(polyline_str)
    
    # Convert the list of coordinates into a DataFrame
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    # Initialize the distance column with zeros
    df['distance'] = 0.0

    def haversine(lat1, lon1, lat2, lon2):
        """
        Calculate the Haversine distance between two points in meters.

        :param lat1: Latitude of the first point
        :param lon1: Longitude of the first point
        :param lat2: Latitude of the second point
        :param lon2: Longitude of the second point
        :return: Distance between the two points in meters
        """
        R = 6371000  # Earth radius in meters
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    # Calculate distances between successive points using the Haversine formula
    for i in range(1, len(df)):
        df.loc[i, 'distance'] = haversine(
            df.loc[i-1, 'latitude'], df.loc[i-1, 'longitude'],
            df.loc[i, 'latitude'], df.loc[i, 'longitude']
        )
    
    return pd.DataFrame(df)  # Return the DataFrame

# Input Below
polyline_str = '_p~iF~ps|U_ulLnnqC_mqNvxq`@'
print(polyline_to_dataframe(polyline_str))

# Question 7
def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    
    n = len(matrix)  # Determine the size of the matrix
    
    # Step 1: Rotating the matrix by 90 degrees clockwise
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    
    # Step 2: creating the final matrix with sums of rows and columns excluding the current element
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i])  # Sum of the entire row
            col_sum = sum(rotated_matrix[k][j] for k in range(n))  # Sum of the entire column
            final_matrix[i][j] = row_sum + col_sum - rotated_matrix[i][j]  # Exclude the current element

    return final_matrix  # Return final_matrix

# Input Below
print(rotate_and_multiply_matrix([[1, 2, 3],[4, 5, 6],[7, 8, 9]]))

# Question 8
def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    
    # Defining the order of the days of the week
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Group by (id, id_2) pair
    grouped = df.groupby(['id', 'id_2'])
    
    # Initialize a boolean series to store the result
    result = pd.Series(index=grouped.indices.keys(), dtype=bool)
    
    # Loop over each group to check for completeness
    for key, group in grouped:
        # Check if all 7 days are covered
        days_covered = set(group['startDay']).union(set(group['endDay']))
        complete_days = set(days_of_week).issubset(days_covered)
        
        # Check if the time range spans the full 24-hour period
        full_time_coverage = (
            (group['startTime'].min() == '00:00:00') and 
            (group['endTime'].max() == '23:59:59')
        )
        
        # Mark True if the time data is incomplete
        result[key] = not (complete_days and full_time_coverage)
    
    # Returned the result as a pd.Series as per instructions
    return pd.Series(result)

# Input Below
print(time_check(df))
