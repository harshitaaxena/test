import streamlit as st


def set_custom_css():
    """Set custom CSS."""
    
    st.markdown(
        """
    <style>
        .block-container {
            max-width: 90%;
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        @media (min-width: 1200px) {
            .reportview-container {
                max-width: 1100px;
            }
        }
        @media (min-width: 992px) and (max-width: 1199.98px) {
            .reportview-container {
                max-width: 930px;
            }
        }
        @media (min-width: 768px) and (max-width: 991.98px) {
            .reportview-container {
                max-width: 720px;
            }
        }
        @media (max-width: 767.98px) {
            .reportview-container {
                max-width: 540px;
            }
        }
        .reportview-container {
            flex-direction: row-reverse;
        }
        .sidebar .sidebar-content {
            padding-left: 10px;
            padding-right: 0;
        }
        .main .block-container {
            padding-left: 0;
            padding-right: 0px;
        }
    </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('')
