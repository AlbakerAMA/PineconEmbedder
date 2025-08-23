@echo off
echo ====================================
echo Installing Document Embedding Dependencies
echo ====================================
echo.

echo Installing core dependencies...
pip install streamlit

echo.
echo Installing document processing libraries...
pip install PyPDF2
pip install python-docx

echo.
echo Installing embedding and vector database libraries...
pip install sentence-transformers
pip install pinecone-client

echo.
echo Installing text processing libraries...
pip install nltk
pip install pandas

echo.
echo ====================================
echo Installation complete!
echo ====================================
echo.
echo Testing installation...
python simple_test.py

echo.
echo You can now run the app with:
echo streamlit run app.py
pause