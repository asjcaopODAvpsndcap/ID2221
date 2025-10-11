Medical RAG Q&A System
This is a Retrieval-Augmented Generation (RAG) project built on a local Large Language Model (LLM).
It processes a large dataset of medical literature in XML format, cleans and vectorizes the data, and finally deploys an intelligent Q&A web application.
Users can ask questions about medical literature through the frontend interface. The system retrieves the most relevant context from the processed data and uses a locally-hosted LLM to generate accurate, reliable answers, complete with citations to the source documents.

✨ Features
End-to-End Data Pipeline: Uses Apache Spark to automatically clean and transform raw XML files into a structured Parquet format.
Efficient Cloud Storage: Leverages Amazon S3 as a data lake for storing both raw and processed data, ensuring high availability and scalability.
Local & Private RAG: The entire RAG pipeline (embedding and generation) runs on locally deployed Ollama models, eliminating reliance on expensive third-party APIs and ensuring data privacy.
Decoupled Architecture: Built with a FastAPI backend and a static HTML/CSS/JS frontend for a clean, maintainable, and scalable architecture.
Intelligent Q&A with Citations: Provides not only intelligent answers but also allows users to trace information back to the original source documents, ensuring credibility.

🛠️ Tech Stack
Data Processing: Apache Spark, Java
Data Storage: Amazon S3
Backend: Python, FastAPI, LangChain
Vector Database: ChromaDB
Local Models: Ollama (deepseek-r1:7b for LLM, embeddinggemma:latest for embeddings)
Frontend: HTML, CSS, JavaScript

🚀 Getting Started
1. Prerequisites
Before you begin, ensure you have the following installed on your system:
Java: JDK 11 or higher (for running Spark)
Python: 3.9 or higher
Git: For cloning the repository
AWS CLI: Configured with your AWS access credentials
Ollama: Visit ollama.com to download and install. After installation, pull the required models:
ollama pull deepseek-r1:7b
ollama pull embeddinggemma:latest

2. Project Setup
Install Python dependencies
pip install -r requirements.txt

Configure Environment Variables
For the Java Spark Job: Create a config.properties file in the src/main/resources directory and add your AWS credentials.


# src/main/resources/config.properties
aws.accessKeyId=YOUR_AWS_ACCESS_KEY_ID
aws.secretAccessKey=YOUR_AWS_SECRET_ACCESS_KEY
s3.endpoint=s3.eu-north-1.amazonaws.com
s3.bucket=s3a://your-s3-bucket-name

# .env
AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_ACCESS_KEY"
AWS_REGION="eu-north-1"

3. Execution Flow
Please follow these steps in order.
Step 1: Data Processing (Run the Spark Job)

This step will process the raw XML files from the medical_xml_data/ directory in your S3 bucket and save the results to the processed_data/ directory.
Upload Data: Ensure your raw XML files are uploaded to the medical_xml_data/ directory in your S3 bucket.

Run the Java Application:
It is recommended to open the project in an IDE (like IntelliJ IDEA).
Make sure the IDE has recognized it as a Maven project and downloaded all dependencies.
Locate the NoSQLProjectMain.java file and run its main method.
Wait for the Spark job to complete. Upon success, a processed_data/ folder should appear in your S3 bucket.

Step 2: Create the Vector Index
This step reads the processed Parquet files from S3, computes embeddings using the local Ollama model, and stores them in a local Chroma database.
Run the following command in the project's root directory:```bash
python build_index.py

Step 4: Access the Frontend UI
Once the server is running, open your browser and navigate to:
http://localhost:8000
