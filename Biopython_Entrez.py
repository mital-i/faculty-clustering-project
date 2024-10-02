# Import required packages
import pandas as pd # Opens and manipulates data frames
from Bio import Entrez # Provides access to NCBI's databases (Uses Bio Python package)
from collections import Counter # Calculates frequencies
import re # Needed to clean strings
import os # Needed for navigating the computer's filepath system
import ssl

# Disable SSL Verification
ssl._create_default_https_context = ssl._create_unverified_context

# Open two data frames: the main one and a search terms column.
faculty_df = pd.read_excel("BioSci Faculty.xlsx")
search_term = pd.read_excel("BioSci Faculty.xlsx", usecols=["Faculty_Author_Affiliation"])
research_keywords = pd.read_excel("Research Keywords - 2024.07.23.xlsx")

# Create a new empty column called 'PMIDs'.
faculty_df["PMIDs"] = None

# Let NCBI know who's accessing database.
Entrez.email = "sarkisj@uci.edu"

# Creates a loop to search for PubMed publications between YYYY and YYYY for each faculty member; adds the PMID for each article in the column called PMIDs. (Can also format mindate and maxdate as YYYY/MM/DD.)
for faculty in range(len(faculty_df)):
    search_term = faculty_df.iloc[faculty]["Faculty_Author_Affiliation"] # Get search terms from Faculty_Author_Affiliation column in data frame.
    handle_search = Entrez.esearch( # Search function
        db = "pubmed", # Use the PubMed database
        mindate = "2019", # Start date for publication search
        maxdate = "2024", # End date for publication search
        term = search_term # Search terms to enter into PubMed
        )
    record = Entrez.read(handle_search)
    idlist = record["IdList"]
    faculty_df.at[faculty, "PMIDs"] = idlist
    handle_search.close()

# Let NCBI know who's accessing their database
Entrez.email = "sarkisj@uci.edu"

# Create loop. For each PMID for each faculty member, retrieve the associated MeSH terms.
for faculty in range(len(faculty_df)):
    PMID_list = faculty_df.iloc[faculty]['PMIDs']
    mesh_term_texts = []
    for PMID in PMID_list:
        handle_mesh = Entrez.efetch(db="pubmed", id=PMID, retmode="xml")
        record = Entrez.read(handle_mesh)
        # Check if "PubmedArticle" list is not empty
        if record["PubmedArticle"]:
            # Access the 'MedlineCitation' part of the record
            medline = record["PubmedArticle"][0]["MedlineCitation"]
            # Access the 'MeshHeadingList' part of the 'MedlineCitation'
            mesh_headings = medline.get("MeshHeadingList", [])
            # Extract the 'DescriptorName' of each 'MeshHeading'
            for mesh_heading in mesh_headings:
                descriptor_name = mesh_heading["DescriptorName"]  # Extract the string value
                mesh_term_texts.append(descriptor_name)  # Add the descriptor_name to the list
        handle_mesh.close()
    # Convert the list of Mesh terms to a string and assign it to the DataFrame cell
    faculty_df.at[faculty, 'Mesh_Terms'] = '; '.join(mesh_term_texts)

# For the mapped MeSH terms, this cell combines the MeSH terms for each faculty member under one column called Mesh_Terms.

# Open data frame that has the mapped MeSH terms.
mapped_mesh_terms = pd.read_excel("research_keywords_cleaned_mesh_terms.xlsx")

#Function to duplicate the items in the Mesh_Terms column
def duplicate_mesh_terms(mesh_terms):
    if pd.notna(mesh_terms) and mesh_terms != "":
        return str(mesh_terms) + "; " + str(mesh_terms)
    return mesh_terms

mapped_mesh_terms['Mesh_Terms'] = mapped_mesh_terms['Mesh_Terms'].apply(duplicate_mesh_terms) # Apply the function to the Mesh_Terms column

# Merge the additional terms with the combined faculty DataFrame
combined_faculty_df = pd.merge(faculty_df, mapped_mesh_terms, left_on='Faculty', right_on='Faculty', how='inner')

# Combine 'Mesh_Terms_x' and 'Mesh_Terms_y' into one column for each 'Faculty'
combined_faculty_df['Combined_Mesh_Terms'] = combined_faculty_df['Mesh_Terms_x'].fillna('') + '; ' + combined_faculty_df['Mesh_Terms_y'].fillna('')

# Rename individual columns to preserve them
combined_faculty_df.rename(columns = {'Mesh_Terms_x': 'Mesh_Terms_Abstracts', 'Mesh_Terms_y': 'Mesh_Terms_Mapped'}, inplace = True)

# This cell removes unhelpful MeSH terms from each faculty member's Combined_Mesh_Terms list.

# List of terms to remove
remove_terms = [
    "Animals",
    "Biology",
    "Humans", 
    "Rats", 
    "Mice", 
    "Male", 
    "Female", 
    "Disease Models, Animal", 
    "Mice, Transgenic",
    "Mice, Inbred C57BL",
    "Mice, Knockout",
    "Adult",
    "Middle Aged",
    "Models, Theoretical",
    "Models, Biological",
    "Models, Animal"
    ]

# Function to remove specified terms from a string of terms
def remove_terms_from_string(terms_string, terms_to_remove):
    terms_list = terms_string.split('; ')
    filtered_terms = [term for term in terms_list if term not in terms_to_remove]
    cleaned_string = '; '.join(filtered_terms).strip('; ')
    cleaned_string = re.sub(r';\s*;', ';', cleaned_string)
    return cleaned_string

# Loop through each faculty member and remove the specified terms from their Mesh_Terms
for index, row in combined_faculty_df.iterrows():
    combined_faculty_df.at[index, 'Combined_Mesh_Terms'] = remove_terms_from_string(row['Combined_Mesh_Terms'], remove_terms)

# This cell counts the top MeSH terms and their frequency proportions.

def find_most_common_item(data):
  # Split the data into a list of items
  items = data.split('; ')

  # Count the occurrences of each item
  item_counts = Counter(items)

  # Find the most common item and its count
  most_common_item, max_count = item_counts.most_common(1)[0]

  # Calculate the average frequency
  average_frequency = sum(item_counts.values()) / len(item_counts)

  return most_common_item, average_frequency

# Initialize the columns
combined_faculty_df['most_common_item'] = None
combined_faculty_df['average_frequency'] = None

for i in range(len(combined_faculty_df)):
  most_common, average_freq = find_most_common_item(combined_faculty_df['Combined_Mesh_Terms'].iloc[i])
  combined_faculty_df.at[i, 'most_common_item'] = most_common
  combined_faculty_df.at[i, 'average_frequency'] = average_freq

# Initialize the counter
counter = Counter()

# Loop through all the strings in the 'Combined_Mesh_Terms' column
for string in combined_faculty_df['Combined_Mesh_Terms']:
    # Convert the string to a list of items and update the counter
    items = str(string).split("; ")
    counter.update(items)

# Calculate the total number of items
total_items = sum(counter.values())

# Get the most common items
top_items = counter.most_common(150)

# Print the most common items along with their proportion
for item, count in top_items:
    proportion = count / total_items
    print(f"{item}: {count} ({proportion:.2%})")

# Function to calculate the normalized scores for a faculty member
def calculate_normalized_scores(faculty_string, top_items):
    faculty_counter = Counter(str(faculty_string).split("; "))
    total_count = sum(faculty_counter.values())
    # Extract the item names from the top_items tuples
    item_names = [item[0] for item in top_items]
    scores = {item: faculty_counter.get(item, 0) / total_count for item in item_names}
    return scores

# Apply the score calculation to each faculty member
combined_faculty_df['Normalized_Scores'] = combined_faculty_df['Combined_Mesh_Terms'].apply(lambda x: calculate_normalized_scores(x, top_items))

# Convert the 'Normalized_Scores' column with dictionaries into separate columns
normalized_scores_df = combined_faculty_df['Normalized_Scores'].apply(pd.Series)

# Add a prefix to the new columns
normalized_scores_df = normalized_scores_df.add_prefix('Normalized_')

# Ensure the indices of both DataFrames match
normalized_scores_df.index = combined_faculty_df.index

# Concatenate the new columns to the original DataFrame
combined_faculty_df = pd.concat([combined_faculty_df, normalized_scores_df], axis=1)

# Revise the existing Excel document.
combined_faculty_df.to_excel('faculty_mesh_terms.xlsx', index = False)

# Define the columns to drop, including "Unnamed: 2" through "Unnamed: 29"
columns_to_drop = ['Faculty_Author', 'Faculty_Author_Affiliation', 'PMIDs', 'Mesh_Terms_Abstracts', 'Mesh_Terms_Mapped', 'Combined_Mesh_Terms', 'Normalized_Scores', 'most_common_item', 'average_frequency'] + [f'Unnamed: {i}' for i in range(2, 30)]

# Drop the specified columns
pca_matrix = combined_faculty_df.drop(columns=columns_to_drop)

# STEP X: Export matrix for PCA analyses
current_folder = os.getcwd() # Get the current working directory

file_path = os.path.join(current_folder, "mesh_terms_matrix_5yrs_and_keywords.xlsx") # Define the file path within the current folder

pca_matrix.to_excel(file_path, index=False) # Save the file to the current folder

#pca_matrix.to_excel('/Users/mitalimittal/Downloads/keywords project/mesh_terms_matrix_5yrs_and_keywords.xlsx', index = False)