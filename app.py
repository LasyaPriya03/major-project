import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter
import transformers
import io
import os
import tempfile

# Streamlit app
def main():
    st.set_page_config(page_title="Major Project", page_icon=":books:")
    st.title("Sample project frontend")
    st.subheader("Upload your document:")

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])

    # Enter button
    if st.button("Enter"):
        if uploaded_file is not None:
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            # Load the document
            loader = UnstructuredFileLoader(temp_file_path)
            data = loader.load()

            # Split into chunks
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(data)

            # Initialize summarization model
            model = transformers.AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-large-lama-cnndm")
            tokenizer = transformers.AutoTokenizer.from_pretrained("mrm8488/t5-large-lama-cnndm")

            # Summarize the document
            summary = ""
            for text in texts:
                input_ids = tokenizer(text.page_content, return_tensors="pt").input_ids
                output = model.generate(input_ids, max_length=130, min_length=30, do_sample=False)
                summary += tokenizer.batch_decode(output, skip_special_tokens=True)[0] + "\n\n"

            # Display the summary
            st.subheader("Summarized Text:")
            st.write(summary)

            # Delete the temporary file
            os.unlink(temp_file_path)
        else:
            st.warning("Please upload a PDF file first.")

if __name__ == "__main__":
    main()
