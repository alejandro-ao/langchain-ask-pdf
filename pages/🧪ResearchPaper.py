import streamlit as st


st.set_page_config(page_title="Plotting Demo", page_icon="🧪")

st.markdown("# Plotting Demo")
st.sidebar.header("Plotting Demo")
# st.write("Check if session state variables persist across pages:", st.session_state)

def upload_pdf():
    st.header("Ask your PDF 💬")

    # If a PDF exists in the session state, display its name and a "Remove PDF" button
    if 'uploaded_pdf' in st.session_state:
        col1, col2 = st.columns([10, 4])  # Creating two columns with different widths
        # col2.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)  # Add a CSS white space to align the button
        col1.write(f"Uploaded file: {st.session_state.uploaded_pdf.name}")

        if col2.button('Remove PDF'):
            del st.session_state.uploaded_pdf
            return None

        return st.session_state.uploaded_pdf
    
    # Otherwise, display the file uploader
    else:
        uploaded_pdf = st.file_uploader("Upload your PDF", type="pdf")

        # If the user uploads a new PDF
        if uploaded_pdf:
            st.session_state.uploaded_pdf = uploaded_pdf
            return uploaded_pdf

        return None


pdf = upload_pdf()