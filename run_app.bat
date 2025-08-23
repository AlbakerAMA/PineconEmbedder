@echo off
echo ====================================
echo Document Embedding Streamlit App
echo ====================================
echo.

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Testing the setup...
python test_embedding.py

echo.
echo Starting Streamlit application...
echo Open your browser and go to http://localhost:8501
echo Press Ctrl+C to stop the application
echo.
streamlit run app.py