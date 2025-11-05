Parallel Text Processing & Sentiment Benchmarking App


This is a full-stack Streamlit web application for high-speed text analysis, featuring a complete user authentication system and parallel processing. It allows users to create secure accounts, upload text data, and benchmark the performance (Speed vs. Accuracy) of classical NLP models (VADER) against modern Large Language Models (OpenAI GPT-3.5).
The app's backend uses Python's multiprocessing library to run heavy NLP tasks on all available CPU cores, and smtplib to send automated email reports.
🚀 Core Features
 * Secure User Portal: Full CRUD authentication (Create, Read, Update, Delete) with sqlite3. Users can sign up, log in, edit profiles, and reset passwords.
 * Parallel Processing Engine: A multiprocessing pool (mp.Pool) distributes text processing tasks to dramatically reduce computation time on large datasets.
 * Sentiment Benchmarking:
   * Classical: vaderSentiment for fast, rule-based analysis.
   * LLM: openai.ChatCompletion (gpt-3.5-turbo) for context-aware analysis.
 * Performance Dashboard: Automatically generates and displays:
   * matplotlib comparison graphs (Time vs.Accuracy).
   * scikit-learn Confusion Matrices (confusion_matrix, ConfusionMatrixDisplay).
   * pandas DataFrame summary tables.
 * Data Processing:
   * Text Cleaning: A custom regex (re) function to clean text.
   * Topic Modeling: TfidfVectorizer and KMeans from scikit-learn to cluster topics.
 * Automated Email Reports: Uses Python's built-in smtplib and email.mime modules to send results (graphs, CSVs) directly to the user's registered email.
⚙ Technology Stack
|---------------|------------------|------------------------------------------------------|
|  Category     |   Technology     |                       Purpose                        |
|---------------|------------------|------------------------------------------------------|
| Frontend      | Streamlit        | Core UI, dashboard, widgets, and file uploader.      |
| Backend       | Python 3.10+     | All backend logic.                                   |
| Parallelism   | multiprocessing  | The core engine for high-speed processing.           |
| Database      | sqlite3          | (via database.py) Storing all user and profile data. |
| User Auth     | (custom auth.py) | (Hashing/verification, likely bcrypt)                |
| Classical NLP | vaderSentiment   | Fast, baseline sentiment analysis.                   |
| Modern NLP    | openai           | LLM-based sentiment (SOTA) analysis.                 |
| ML & Metrics  | scikit-learn     | TfidfVectorizer, KMeans, accuracy_score.             |
| Data Handling | pandas           | Data loading, manipulation, and CSV creation.        |
| Visualization | matplotlib       | Generating all performance graphs.                   |
| Notifications | smtplib          | Sending automated email reports via Gmail.           |
| Utilities     | python-dotenv    | Managing environment variables (.env).               |
|               |  zipfile         |  Compressing results for download.                   |
|---------------|------------------|------------------------------------------------------|

The app will open in your browser. You can now create an account, log in, and start processing.
🏗 Project Structure
.
├── .venv/                 # Virtual environment
├── images/                # Background images (e.g., home.jpg)
├── outputs/               # Saved results (graphs, CSVs)
├── _pycache_/           # Python cache (add to .gitignore)
├── .env                   # <-- YOUR SECRETS (MUST BE IN .gitignore)
├── .gitignore             # Git ignore file
├── app.py                 # <-- MAIN STREAMLIT APP
├── auth.py                # User authentication logic
├── database.py            # SQLite database setup and user functions
└── requirements.txt       # Project dependencies

🏃‍♂ Installation & Usage
1. Prerequisites
Python 3.10+
Git
An OpenAI API Key
A Gmail "App Password" (for the email sender)
2. Clone & Setup
# Clone the repository
git clone https://github.com/[YourUsername]/[YourRepositoryName].git
cd [YourRepositoryName]

# Create and activate a virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate.ps1

# Activate (Mac/Linux)
source .venv/bin/activate
3. Install Dependencies
This project requires a requirements.txt file.
requirements.txt:
streamlit
python-dotenv
pandas
vaderSentiment
scikit-learn
matplotlib
openai
# Add 'bcrypt' if your auth.py uses it for hashing
Run the installer:
pip install -r requirements.txt
4. Configure Secrets (.env)
Create a file named .env in the root of the project. This file is critical and should NOT be committed to GitHub.
# .env

# 1. OpenAI API Key (for LLM sentiment)
OPENAI_API_KEY="sk-YourSecretOpenAIKey"

# 2. Gmail Sender Email (for sending reports)
EMAIL_SENDER="your-email@gmail.com"

# 3. Gmail App Password (NOT your regular password)
EMAIL_APP_PASSWORD="your_16_digit_app_password"
5. Run the App
With your virtual environment active and your .env file in place, run the app:
streamlit run app.py

Use the App:
Create an account using the "Signup" page.
Log in and go to the "Dashboard".
Upload your data and run the processing.

📈 Potential Improvements
Integrate Local LLMs: Add support for local models (e.g., via Hugging Face) to remove the OpenAI API dependency.
Database Upgrade: Migrate from SQLite to PostgreSQL for better scalability in a multi-user environment.
Asynchronous Tasks: Move the email and processing tasks to a background worker (like Celery) so the UI doesn't block.
Advanced Metrics: Add Precision, Recall, and F1-score to the comparison dashboard.
UI Polish: Allow users to tune K-Means parameters (e.g., number of clusters) from the UI.
