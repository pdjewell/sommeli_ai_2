import numpy as np
import pandas as pd
import os
from PIL import Image
import streamlit as st 
from streamlit import components
from datasets import Dataset, load_dataset, load_from_disk
import faiss
from scripts.preprocessing import preprocess

# App config 
icon = Image.open('./images/wine_icon.png')
st.set_page_config(page_title="Sommeli-AI", 
                   page_icon=icon,
                   layout="wide")
hide_default_format = """
       <style>
       #MainMenu {visibility: visible; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

# App functions 
@st.cache_data 
def read_data(ds_path=None):

    if ds_path is not None:
        # Read in hf file 
        embeddings_dataset = load_from_disk(ds_path)
    else:
        embeddings_dataset = load_dataset("pdjewell/sommeli_ai", split="train")

    # Convert to pandas df
    embeddings_dataset.set_format("pandas")
    df = embeddings_dataset[:]

    # preprocess data (add type col, remove dups)
    df = preprocess(df)
  
    return df 


def get_neighbours(df, query_embedding, k=6,
                   metric='inner'):

    # convert from pandas df to hf ds 
    ds = Dataset.from_pandas(df)
    ds.reset_format()
    ds = ds.with_format("np")

    # add faiss index
    if metric == 'inner':
        ds.add_faiss_index(column="embeddings",
                       metric_type=faiss.METRIC_INNER_PRODUCT)
    else: 
        ds.add_faiss_index(column="embeddings",
                       metric_type=faiss.METRIC_L2)
        
    scores, samples = ds.get_nearest_examples(
        "embeddings", query_embedding, k=k)
    
    samples.pop('embeddings')
    samples.pop('__index_level_0__')
    
    return scores, samples


def filter_df_search(df: pd.DataFrame) -> pd.DataFrame:
    
    modify_search = st.checkbox("🔍 Further filter search selection")
    
    if not modify_search:
        return df

    df = df.copy()

    modification_container_search = st.container()

    with modification_container_search:
        to_filter_columns = st.multiselect("Filter on:", 
                                           ['Province', 'Region', 'Winery','Score', 'Price'],
                                           key='search')

        for column in to_filter_columns:
            if column in ['Score', 'Price']: # Use slider for 'points' and 'price'
                min_val = 0
                max_val = int(df[column].max())
                user_input = st.slider(f"Values for {column}", min_val, max_val, (min_val, max_val))
                df = df[(df[column] >= user_input[0]) & (df[column] <= user_input[1])]
            elif column in ['Country', 'Province', 'Region', 'Variety', 'Winery']: # Use multiselect for these columns
                unique_values = df[column].dropna().unique()
                default_values = [unique_values[0]] if len(unique_values) > 0 else [] # Select only the first unique value if it exists
                user_input = st.multiselect(f"Values for {column}", unique_values, default_values)
                df = df[df[column].isin(user_input)]

    return df


def filter_df_recs(df: pd.DataFrame) -> pd.DataFrame:
    
    modify_recs = st.checkbox("🔍 Filter recommendation results")
    
    if not modify_recs:
        return df

    df = df.copy()

    modification_container_recs = st.container()

    with modification_container_recs:

        to_filter_columns2 = st.multiselect("Filter on:", 
                                            ['Country','Province', 'Region', 'Variety', 'Winery',
                                             'Score', 'Price'],
                                            key='recs')

        for column in to_filter_columns2:
            if column in ['Score', 'Price']: # Use slider for 'points' and 'price'
                min_val = 0
                max_val = int(df[column].max())
                user_input = st.slider(f"Values for {column}", min_val, max_val, (min_val, max_val))
                df = df[(df[column] >= user_input[0]) & (df[column] <= user_input[1])]
            elif column in ['Country', 'Province', 'Region', 'Variety', 'Winery']: # Use multiselect for these columns
                unique_values = df[column].dropna().unique()
                default_values = [unique_values[0]] if len(unique_values) > 0 else [] # Select only the first unique value if it exists
                user_input = st.multiselect(f"Values for {column}", unique_values, default_values)
                df = df[df[column].isin(user_input)]

    return df


if __name__ == "__main__":
    st.title("🍷 Sommeli-AI")
    col1, col2 = st.columns([0.6,0.4], gap="medium")
 
    # Read in data 
    ds_path = "./data/wine_ds.hf"
    df = read_data(ds_path=None)

    with col2:
        st.header("Explore the world of wine  🌍")
        wine_plot = st.radio('Select plot type:', ['2D','3D'],
                            label_visibility = "hidden",
                            horizontal=True)
        st.text("Click the legend categories to filter")

        # Load the HTML file
        with open('./images/px_2d.html', 'r') as file:
            plot2d_html = file.read()
        # Load the HTML file
        with open('./images/px_3d.html', 'r') as file:
            plot3d_html = file.read()
        # Display the HTML plot in the Streamlit app
        if wine_plot == '2D':
            components.v1.html(plot2d_html, width=512, height=512)
        elif wine_plot == '3D': 
            components.v1.html(plot3d_html, width=512, height=512)

    with col1: 

        # Select all wine types initially
        st.header("Search for similar wines  🥂")
        # Select wine type: default is all 
        wine_types = df['Type'].unique()
        selected_wine_types = st.multiselect("Select category 👇", wine_types, default=wine_types)
        df = df[df['Type'].isin(selected_wine_types)]
        subcol1, subcol2 = st.columns([0.5,0.5], gap="small")
        with subcol1:
            # Select wine variety: default is all 
            wine_vars = df['Variety'].unique()
            selected_wine_vars = st.multiselect("Narrow down the variety 🍇",['Select all'] + list(wine_vars),
                                                default = 'Select all')
            if "Select all" in selected_wine_vars:
                df_search = df
            else:
                df_search = df[df['Variety'].isin(selected_wine_vars)]
        
        with subcol2:
            # Select the country: default is all 
            countries = df_search['Country'].unique()
            selected_countries = st.multiselect("Narrow down the country 🌎",['Select all'] + list(countries),
                                                default = 'Select all')
            if "Select all" in selected_countries:
                df_search = df_search
            else:
                df_search = df_search[df_search['Country'].isin(selected_countries)]

        # Add additional filters 
        df_search = filter_df_search(df_search)

        # Create a search bar for the wine 'title'
        selected_wine = st.selectbox("Search for and select a wine 👇", [''] + list(df_search["Title"].unique()))
    
        if selected_wine:
            # Get the embedding for selected_wine
            query_embedding = df.loc[df['Title']==selected_wine, 'embeddings'].iloc[0]

            tasting_notes = df.loc[df['Title']==selected_wine, 'Tasting notes'].iloc[0]
            st.write(f"Tasting notes: {tasting_notes}")

            # CSS to inject contained in a string
            hide_table_row_index = """
                        <style>
                        thead tr th:first-child {display:none}
                        tbody th {display:none}
                        </style>
                        """
            # Inject CSS with Markdown
            st.markdown(hide_table_row_index, unsafe_allow_html=True)

            # Display selected wine
            st.header("	🍷 Your selected wine")
            selected_cols = ['Title','Country','Province','Region','Winery',
                            'Variety','Tasting notes','Score']
            st.table(df.loc[df['Title']==selected_wine, selected_cols].fillna(""))

            # Slider for results to show 
            k = st.slider(f"Choose how many similar wines to show 👇", 1, 10, value=4)
            
            # Filter recommendation results 
            df_results = filter_df_recs(df)
            
            # Display results as table 
            if st.button("🔘 Press me to generate similar tasting wines"):
                # Get neighbours
                scores, samples = get_neighbours(df_results, query_embedding, 
                                                 k=k+1, metric='l2')
                recs_df = pd.DataFrame(samples).fillna("")
                recs_df = recs_df.fillna(" ")
                # Display results
                st.header(f"🍾 Top {k} similar tasting wines")
                st.table(recs_df.loc[1:,selected_cols])
            
        else:
            print("Awaiting selection")
