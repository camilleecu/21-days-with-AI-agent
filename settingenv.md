# Creating the project
create a virtual environemnt


First, install uv (a fast Python package manager).
in terminal:  pip install uv


Then create an empty folder for the project and initialize it:

mkdir llm-zoomcamp-code
cd llm-zoomcamp-code
uv init

this create a file "pyproject.toml"

add "[tool.uv]
env = "llmzoomcamp"" to "pyproject.toml" 

Now add the dependencies we'll need:

uv add requests minsearch openai jupyter python-dotenv


This installs:

requests - to fetch the FAQ dataset from the internet
minsearch - a simple in-memory search engine for indexing and searching text
openai - the OpenAI API client for calling the LLM
jupyter - the notebook environment where we'll write and run code
python-dotenv - to load API keys from a .env file




use notebooks directly inside VS Code (even in Codespaces). The goal is: let VS Code use your llmzoomcamp env as kernel, no browser needed.


by the end of the dat
# 1. Check what changed
git status

# 2. Add all changes
git add .

# 3. Commit with a short message
git commit -m "day1 notebook + env setup"

# 4. Push to GitHub remote
git push