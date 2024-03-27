import streamlit as st
from apriori import render as render_apriori
from d_trees import render as render_dtrees

def main():
    
    st.sidebar.title('Treatment Recommendation with Jusitfication')

    page = st.sidebar.selectbox(
        "Select a technique",
        ["Apriori","Interpretable Decision Trees"]
    )

    # Render the selected page
    if page == "Apriori":
        render_apriori()
    elif page == "Interpretable Decision Trees":
        render_dtrees()

if __name__ == "__main__":
    main()
