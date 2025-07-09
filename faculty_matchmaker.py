import streamlit as st
import pandas as pd

# Set up the page layout
st.set_page_config(page_title="Faculty Research Matchmaker", layout="wide")

# Add title and description
st.title("Faculty Research Matchmaker")
st.markdown("""
This tool matches funding opportunities with faculty members.
"""
)

st.divider()

st.subheader("How to Use This Tool")
st.markdown("1. Enter Enter MeSH terms from a funding opportunity (separated by semicolons).")
with st.expander("**What are MeSH terms?**"):
    st.markdown('''
        MeSH terms, or Medical Subject Headings, are standardized keywords used to index and categorize biomedical literature. They help in identifying relevant research topics and areas of expertise.
    ''')
    st.markdown("""
        You can use the **MeSH on Demand Tool** to extract MeSH terms from the text of a funding opportunity.
        Simply paste the text into the tool, and it will return a list of relevant MeSH terms.
    """)
    st.link_button(label="**Use MeSH on Demand**", url="https://meshb.nlm.nih.gov/MeSHonDemand", type="secondary")

st.markdown("2. The tool uses Jaccard similarity to determine the overlap with faculty research interests.")
st.markdown("3. Results show ranked faculty members with matching scores and overlapping terms.")

st.divider()

# Import the faculty mesh terms from the Excel file
faculty_mesh_terms_df = pd.read_excel("/Users/sarkisj/Library/CloudStorage/OneDrive-UCIrvine/BioSci Research Development/Faculty-Keyword-Inventory-Project/faculty-mapped-mesh-terms/faculty_unique_mesh_terms.xlsx")

# Create a dictionary with faculty names as keys and their unique MeSH terms as values
faculty_terms = faculty_mesh_terms_df.set_index('Faculty_Full_Name')['Unique_Mesh_Terms'].to_dict()

st.header("Use the Tool")

# Import the funding opportunities mesh terms from the Excel file
user_input = st.text_input("**Enter the MeSH terms of the funding opportunity (separated by semicolons):**")
funding_terms = user_input
if not user_input:
    st.warning("Enter MeSH terms to continue.")
    st.stop()

@st.cache_data
def smart_title(text):
    exceptions = {"and", "or", "on", "the", "a", "an", "in", "of", "to", "with", "for", "by", "over"}
    words = text.strip().split()
    if not words:
        return ''
    titled = [words[0].capitalize()]
    for word in words[1:]:
        if word.lower() in exceptions:
            titled.append(word.lower())
        else:
            titled.append(word.capitalize())
    return ' '.join(titled)

funding_terms = {smart_title(term) for term in funding_terms.split(';')}

# # Access the dictionary with a specific faculty name
# user_input = st.text_input("Enter faculty member name:")
# faculty_name = user_input
# if faculty_name in faculty_terms:
#     f"{faculty_terms[faculty_name]}"
# else:
#     f"We cannot find {faculty_name} in the dictionary. Be sure to enter the full name in the format _Last, First_."

# Function to calculate Jaccard similarity between two sets
@st.cache_data
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0.0
    return round((intersection / union) * 100, 2)

# Function to predict faculty fit based on MeSH terms
@st.cache_data
def predict_faculty_fit(faculty_data, funding_terms):
    """
    Predicts faculty fit for a funding opportunity based on MeSH terms.

    Args:
        faculty_data (dict): Dictionary of faculty IDs and their MeSH term lists (sets).
        funding_terms (set): Set of MeSH terms for the funding opportunity.

    Returns:
        list: A list of tuples containing (faculty_id, similarity_score, overlapping_terms)
              sorted in descending order of similarity.
    """
    faculty_scores = []
    for faculty_id, faculty_terms in faculty_data.items():
        overlap = set(faculty_terms).intersection(funding_terms)
        similarity = jaccard_similarity(set(faculty_terms), funding_terms)
        faculty_scores.append((faculty_id, similarity, overlap))

    # Sort by similarity score in descending order
    ranked_faculty = sorted(faculty_scores, key=lambda x: x[1], reverse=True)
    return ranked_faculty

# Convert faculty terms to sets, skipping any null values
faculty_mesh_sets = {}
for k, v in faculty_terms.items():
    if pd.notna(v):  # Check if the value is not NaN
        # Split the string by semicolon and strip whitespace
        terms = set(term.strip() for term in v.split(';'))
        faculty_mesh_sets[k] = terms

ranked_results = predict_faculty_fit(faculty_mesh_sets, set(funding_terms))

# Create a dataframe from the results
results_df = pd.DataFrame(
    [(faculty, f"{score:.4f}", ", ".join(sorted(overlap))) 
     for faculty, score, overlap in ranked_results if score > 0],
    columns=['Faculty', 'Jaccard Percentage', 'Overlapping Terms']
)

if not results_df.empty:
    st.subheader("Ranked Faculty by Fit")
    st.markdown(
"""
The following faculty members are ranked based on their fit with the provided MeSH terms. The score indicates the Jaccard similarity, and overlapping terms are listed for each faculty member.
"""
)
    st.dataframe(results_df, hide_index=True, use_container_width=True)
else:
    st.warning("No matching faculty found. Make sure that you've entered valid MeSH terms and that they're separated by semicolons.")