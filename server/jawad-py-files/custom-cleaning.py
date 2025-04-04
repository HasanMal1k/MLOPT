def main():
    st.title("CSV Data Transformation Tool")
    
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Data")
        st.dataframe(df.head())
        
        st.write("### Data Type Selection")
        new_dtypes = {}
        for col in df.columns:
            dtype = df[col].dtype
            new_dtypes[col] = st.selectbox(f"{col}",
                                           ["int", "float", "string", "datetime"],
                                           index=["int", "float", "string", "datetime"].index(dtype.name if dtype.name in ["int", "float", "string", "datetime"] else "string"))
        
        # Convert columns to selected data types
        for col, dtype in new_dtypes.items():
            if dtype == "int":
                df[col] = pd.to_numeric(df[col], errors='coerce', downcast='integer')
            elif dtype == "float":
                df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')
            elif dtype == "datetime":
                df[col] = pd.to_datetime(df[col], errors='coerce')
            else:
                df[col] = df[col].astype(str)
        
        st.write("### Column Transformation")
        transformations = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            transformations[col] = st.selectbox(f"Transformation for {col}",
                                                ["None", "Log", "Square Root", "Square", "Cube"], index=0)
        
        # Apply transformations
        for col, trans in transformations.items():
            if trans != "None":
                df[col] = transform_column(df[col], trans)
        
        st.write("### Transformed Data Preview")
        st.dataframe(df.head())
        
        # Download transformed data
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        st.download_button("Download Transformed CSV", buffer, file_name="transformed_data.csv", mime="text/csv")

if _name_ == "_main_":
    main()