# Load required libraries
import openpyxl
import pandas as pd
from Bio import Entrez
from collections import Counter
import re
import time

# Load dataframes
faculty_df = pd.read_excel('biosci_faculty.xlsx', sheet_name='minus_teaching')
research_keywords_df = pd.read_excel('research_keywords.xlsx')
faculty_proposal_mesh_terms_df = pd.read_excel('faculty_proposal_abstracts.xlsx', sheet_name='proposal_abstracts_sheet')
mapped_mesh_terms_df = pd.read_excel('research_keywords_cleaned_mesh_terms.xlsx', usecols=['Faculty_Full_Name', 'Mapped_Mesh_Terms'])

# Collect MeSH terms from publications
# Set Entrez email
Entrez.email = "sarkisj@uci.edu"
# Fetch PMIDs for each faculty member
faculty_df["pmids"] = None
for index, row in faculty_df.iterrows():
    search_term = row["Faculty_Author_Affiliation"]
    handle_search = Entrez.esearch(db="pubmed", mindate="2022", maxdate="2025", term=search_term)
    record = Entrez.read(handle_search)
    faculty_df.at[index, "pmids"] = record["IdList"]
    handle_search.close()
# Fetch MeSH terms for each PMID
faculty_df['pub_mesh_terms'] = None
for index, row in faculty_df.iterrows():
    pmid_list = row['pmids'] 
    mesh_term_texts = []
    for pmid in pmid_list:
        handle_mesh = Entrez.efetch(db="pubmed", id=pmid, rettype="xml")
        record = Entrez.read(handle_mesh)
        if record["PubmedArticle"]:
            medline = record["PubmedArticle"][0]["MedlineCitation"]
            mesh_headings = medline.get("MeshHeadingList", [])
            for mesh_heading in mesh_headings:
                descriptor_name_element = mesh_heading.get("DescriptorName")
                if descriptor_name_element:
                    descriptor_name = str(descriptor_name_element)  # Convert StringElement to string
                    mesh_term_texts.append(descriptor_name)
        handle_mesh.close()
        time.sleep(0.5)
    faculty_df.at[index, 'pub_mesh_terms'] = '; '.join(mesh_term_texts)
# Save the updated DataFrame to a CSV file
output_file = 'faculty_pulled_mesh_terms.csv'
faculty_df.to_csv(output_file, index=False)
# Load the saved CSV file as faculty_df
output_file = 'faculty_pulled_mesh_terms.csv'
faculty_df = pd.read_csv(output_file)

# Collect MeSH terms from proposal abstracts
# Process proposal MeSH terms
faculty_proposal_mesh_terms_df['Proposal_Mesh_Terms'] = faculty_proposal_mesh_terms_df['Proposal_Mesh_Terms'].astype(str)
proposal_mesh_terms_df = faculty_proposal_mesh_terms_df.groupby('Faculty_Full_Name')['Proposal_Mesh_Terms'].agg(lambda x: '; '.join(x)).reset_index()
def repeat_mesh_terms(mesh_terms, repetitions):
    if pd.notna(mesh_terms) and mesh_terms != "":
        repeated_terms = "; ".join([str(mesh_terms)] * repetitions)
        return repeated_terms
    return mesh_terms

# Weight MeSH terms
# Apply doubling
mapped_mesh_terms_df['Mapped_Mesh_Terms'] = mapped_mesh_terms_df['Mapped_Mesh_Terms'].apply(lambda x: repeat_mesh_terms(x, 2))
# Apply tripling
proposal_mesh_terms_df['Proposal_Mesh_Terms'] = proposal_mesh_terms_df['Proposal_Mesh_Terms'].apply(lambda x: repeat_mesh_terms(x, 3))

# Merge dataframes
merged_df = pd.merge(faculty_df, proposal_mesh_terms_df, on='Faculty_Full_Name', how='left')
merged_df.drop(columns=['Faculty', 'Faculty_Author', 'Faculty_Author_Affiliation'], inplace=True)
combined_faculty_df = pd.merge(merged_df, mapped_mesh_terms_df, on="Faculty_Full_Name", how='left')

# Combine MeSH terms
# First, replace NaN values with empty strings
combined_faculty_df['Proposal_Mesh_Terms'] = combined_faculty_df['Proposal_Mesh_Terms'].fillna('')
combined_faculty_df['Mapped_Mesh_Terms'] = combined_faculty_df['Mapped_Mesh_Terms'].fillna('')
combined_faculty_df['pub_mesh_terms'] = combined_faculty_df['pub_mesh_terms'].fillna('')

# Create a function to combine terms
def combine_mesh_terms(row):
    terms = []
    if row['Proposal_Mesh_Terms'] and row['Proposal_Mesh_Terms'].strip():
        terms.append(row['Proposal_Mesh_Terms'])
    if row['Mapped_Mesh_Terms'] and row['Mapped_Mesh_Terms'].strip():
        terms.append(row['Mapped_Mesh_Terms'])
    if row['pub_mesh_terms'] and row['pub_mesh_terms'].strip():
        terms.append(row['pub_mesh_terms'])
    return '; '.join(terms)

# Apply the function to create the combined column
combined_faculty_df['Combined_Mesh_Terms'] = combined_faculty_df.apply(combine_mesh_terms, axis=1)

# Drop original columns if needed
combined_faculty_df.drop(columns=['Proposal_Mesh_Terms', 'Mapped_Mesh_Terms', 'pmids', 'pub_mesh_terms'], inplace=True)

# Remove unhelpful MeSH terms
remove_terms = [
    "Animals", "Biology", "Humans", "Rats", "Mice", "Male", "Female",
    "Disease Models, Animal", "Mice, Transgenic", "Mice, Inbred C57BL",
    "Mice, Knockout", "Adult", "Middle Aged", "Models, Theoretical",
    "Models, Biological", "Models, Animal", "Mentors", "Mentoring",
    "Universities", "Drosophila", "Mammals", "Faculty", "Research Design",
    "Follow-Up Studies", "United States", "Goals", "Preliminary Data", "Students", "Feedback"
]

def remove_terms_from_string(terms_string, terms_to_remove):
    terms_list = terms_string.split('; ')
    filtered_terms = [term for term in terms_list if term not in terms_to_remove]
    cleaned_string = '; '.join(filtered_terms).strip('; ')
    cleaned_string = re.sub(r';\s*;', ';', cleaned_string)
    return cleaned_string

combined_faculty_df['Combined_Mesh_Terms'] = combined_faculty_df['Combined_Mesh_Terms'].apply(lambda x: remove_terms_from_string(x, remove_terms))

# Calculate most common item and average frequency
def find_most_common_item(data):
    items = data.split('; ')
    item_counts = Counter(items)
    if not item_counts:
        return None, 0
    most_common_item, max_count = item_counts.most_common(1)[0]
    average_frequency = sum(item_counts.values()) / len(item_counts)
    return most_common_item, average_frequency

combined_faculty_df['most_common_item'] = None
combined_faculty_df['average_frequency'] = None

for index, row in combined_faculty_df.iterrows():
    most_common, average_freq = find_most_common_item(row['Combined_Mesh_Terms'])
    combined_faculty_df.at[index, 'most_common_item'] = most_common
    combined_faculty_df.at[index, 'average_frequency'] = average_freq

# Count top MeSH terms
counter = Counter()
for string in combined_faculty_df['Combined_Mesh_Terms']:
    items = str(string).split("; ")
    counter.update(items)

total_items = sum(counter.values())
top_items = counter.most_common(150)

for item, count in top_items:
    proportion = count / total_items
    print(f"{item}: {count} ({proportion:.2%})")

# Calculate normalized scores
def calculate_normalized_scores(faculty_string, top_items):
    faculty_counter = Counter(str(faculty_string).split("; "))
    total_count = sum(faculty_counter.values())
    item_names = [item[0] for item in top_items]
    scores = {item: faculty_counter.get(item, 0) / total_count for item in item_names}
    return scores

combined_faculty_df['Normalized_Scores'] = combined_faculty_df['Combined_Mesh_Terms'].apply(lambda x: calculate_normalized_scores(x, top_items))

# Calculate top MeSH terms for each faculty
def calculate_top_mesh_terms(faculty_mesh_terms_dict):
    for faculty, mesh_terms in faculty_mesh_terms_dict.items():
        if mesh_terms:
            term_counts = Counter(mesh_terms)
            top_terms = term_counts.most_common(5)
            print(f"{faculty}: ", end="")
            for i, (term, count) in enumerate(top_terms):
                print(f"{term}", end="")
            print()
        else:
            print(f"No MeSH terms found for {faculty}.")

combined_faculty_df['Mesh_Terms_List'] = combined_faculty_df['Combined_Mesh_Terms'].str.split(';')
combined_faculty_df['Mesh_Terms_List'] = combined_faculty_df['Mesh_Terms_List'].apply(lambda x: [item.strip() for item in x] if isinstance(x, list) else [])

# Create a dictionary with Faculty_Full_Name as keys and Mesh_Terms_List as values
faculty_mesh_terms_dict = dict(zip(combined_faculty_df['Faculty_Full_Name'], combined_faculty_df['Mesh_Terms_List']))
calculate_top_mesh_terms(faculty_mesh_terms_dict)

def get_unique_terms(combined_faculty_df):
    unique_terms = {}
    for index, row in combined_faculty_df.iterrows():
        faculty_name, terms = row['Faculty_Full_Name'], row['Combined_Mesh_Terms']
        terms_list = [term.strip() for term in terms.split(';')]
        unique_terms[faculty_name] = "; ".join(sorted(set(terms_list)))  # Sort for consistency

    return unique_terms

# Get unique MeSH terms
unique_mesh_terms = get_unique_terms(combined_faculty_df)

# Create and save DataFrame of unique terms
unique_terms_df = pd.DataFrame(list(unique_mesh_terms.items()), columns=['Faculty_Full_Name', 'Unique_Mesh_Terms'])
unique_terms_df.to_excel('faculty_unique_mesh_terms.xlsx', index=False)

# Convert 'Normalized_Scores' dictionary column to separate columns
normalized_scores_df = combined_faculty_df['Normalized_Scores'].apply(pd.Series)

# Add prefix to new columns
normalized_scores_df = normalized_scores_df.add_prefix('Normalized_')

# Align indices
normalized_scores_df.index = combined_faculty_df.index

# Concatenate normalized scores to original DataFrame
combined_faculty_df = pd.concat([combined_faculty_df, normalized_scores_df], axis=1)

# Save the updated DataFrame to Excel
combined_faculty_df.to_excel('faculty_mesh_terms.xlsx', index=False)

# Prepare PCA matrix
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# Drop columns to prepare for PCA
pca_matrix = combined_faculty_df.drop(combined_faculty_df.columns[1:36], axis=1)

# Save PCA matrix to Excel
pca_matrix.to_excel('mesh_terms_matrix_5yrs_and_keywords.xlsx', index=False)