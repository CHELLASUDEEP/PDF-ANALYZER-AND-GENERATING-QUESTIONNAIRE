# PDF Analyzer

The PDF Analyzer is a Python application designed to extract text from PDF files, generate questions based on the extracted text, and provide answers to those questions using AI-powered language models.

# NOTE
1. The generated question and answers are stored in a csv file which can be found in output folder.
2. The given pdf is found in static folder.

## Features:

- **PDF Upload**: Users can upload PDF files directly to the application.
- **Text Extraction**: The application extracts text from the uploaded PDF files using PyPDF2 library.
- **Question Generation**: Questions are generated based on the extracted text using OpenAI's GPT-3.5 language model. These questions aim to prepare coders or programmers for exams and coding tests.
- **Answer Generation**: Answers to the generated questions are provided using AI language models. The application utilizes a retrieval-based question answering system to fetch answers from the extracted text.
- **CSV Export**: The generated questions and their corresponding answers are saved to a CSV file for further analysis or use.

## Execution:

1. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

2. Set up your OpenAI API key by replacing `"YOUR_OPENAI_API_KEY"` in the `llm_pipeline` function with your actual API key.

3. Run the application:

    ```bash
    streamlit run main.py
    ```
    write the above line in your terminal for execution.

4. Access the application in your web browser at `http://localhost:8501`.

## Usage:

1. Upload a PDF file using the file uploader.
2. Wait for the analysis to complete.
3. Once the analysis is finished, the application will display the path to the CSV file containing the generated questions and answers.
4. Download the CSV file for further use.



