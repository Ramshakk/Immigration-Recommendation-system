# recommender

Visa Recommendation Service
Table of Contents
•	Installation
•	Configuration
•	Usage
•	Endpoints
•	Functions
•	Error Handling
Installation
1. Clone the repository:
    git clone <repository_url>
2. Navigate to the project directory:
    cd <project_directory>
3. Create and activate a virtual environment:
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
4. Install the required dependencies:
    pip install -r requirements.txt
Configuration
1. Set the OpenAI API key as an environment variable:
    export OPENAI_API_KEY='your_openai_api_key'
    On Windows, use:
set OPENAI_API_KEY='your_openai_api_key'
Usage
1. Run the Flask application:
    python recommender.py
2. The service will be available at http://0.0.0.0:5000.
3. Add /recommend_visa/clwkickdu000013k9ib8helwu for the endpoint
Endpoints
•	/recommend_visa/<id>:
o	Method: POST
o	Input: JSON data containing question-answer pairs
o	Output: Structured visa recommendation data
