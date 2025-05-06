from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import CallbackManager
import shutil
from github import Github
from git import Repo
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph
from langgraph.graph import MessagesState
import os
import dotenv
import time
import google.generativeai as genai
from astrapy import DataAPIClient

dotenv.load_dotenv()

github_token = os.environ.get("GITHUB_TOKEN")
REPO_URL = os.environ.get("REPO_URL")

endpoint = os.environ.get("ASTRA_DB_API_ENDPOINT")
token = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
# If token is not found in environment, use hardcoded value
print("Endpoint:", endpoint)
print("Token:", token[:6], "...")  # Only print the first few chars for security

client = DataAPIClient(token)
db = client.get_database_by_api_endpoint(
  os.environ.get("ASTRA_DB_API_ENDPOINT")
)

print(f"Connected to Astra DB: {db.list_collection_names()}")

class CodeSolution(BaseModel):
    description: str = Field(description="A description of the code solution")
    code: str = Field(description="The code solution")

class CodeReviewer(BaseModel):
    def __init__(self):
        self.github = Github(github_token)
        self.repo = self.github.get_repo("langchain-ai/langchain")

class GraphState(MessagesState):
    state: CodeSolution
    task_context: str
    repo_dir: str
    generated_code: str
    iteration: int

def clone_repo(repo_dir: str, repo_url: str):
    try:
        # if folder exists, delete it
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
    except Exception as e:
        print(f"Error deleting folder {repo_dir}: {e}")
    try:
        Repo.clone_from(f"https://github.com/{REPO_URL}.git", repo_dir)
        print(f"Cloned repo {repo_url} to {repo_dir}")
    except Exception as e:
        print(f"Error cloning repo {repo_url}: {e}")
    return repo_dir

def read_task_context(repo_dir: str):
    with open(os.path.join("/home/system/projects/github-agent", "task.md"), "r") as f:
        return f.read()

def generate_code(task_context: str):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("models/gemini-2.0-flash")
    prompt = f"""You are a senior software engineer. Write clean, efficient, and well-documented Python code for the following task:\n\n{task_context}"""
    response = model.generate_content(prompt)
    return response.text

def initialize_graph():
    graph = StateGraph(GraphState)
    graph.add_node("clone_repo", clone_repo)
    graph.add_node("read_task_context", read_task_context)
    graph.add_node("generate_code", generate_code)
    return graph

def commit_and_push(repo_dir: str, code: str):
    """Write code to solution.py, commit, and push to remote repo."""
    file_path = os.path.join(repo_dir, "solution.py")
    with open(file_path, "w") as f:
        f.write(code)
    repo = Repo(repo_dir)
    repo.git.add("solution.py")
    if repo.is_dirty():
        repo.index.commit("Add/update solution.py with generated code")
        origin = repo.remote(name="origin")
        origin.push()

def search_commit(repo_dir, keyword):
    repo = Repo(repo_dir)
    for commit in repo.iter_commits():
        if keyword in commit.message:
            print(f"Commit: {commit.hexsha}\nMessage: {commit.message}\nDate: {commit.committed_datetime}\n")
    print("Search complete.")

def is_task_complete(repo_dir):
    solution_path = os.path.join(repo_dir, "solution.py")
    if os.path.exists(solution_path):
        print("Task is complete: solution.py exists.")
        return True
    else:
        print("Task is not complete: solution.py does not exist.")
        return False

def task_already_ran(repo_dir, task_context):
    repo = Repo(repo_dir)
    for commit in repo.iter_commits():
        if task_context.strip() in commit.message:
            print("Task already ran (found in commit message).")
            return True
    print("Task not found in commit history.")
    return False

def get_last_solution(repo_dir):
    solution_path = os.path.join(repo_dir, "solution.py")
    if os.path.exists(solution_path):
        with open(solution_path, "r") as f:
            print(f.read())
    else:
        print("No solution.py found.")

def save_to_astra(request, solution, result, commit):
    doc = {
        "request": request,
        "solution": solution,
        "result": result,
        "commit": commit
    }
    collection.insert_one(doc)
    print("Saved to Astra DB.")

def run_and_save(repo_dir):
    task_content = read_task_context(repo_dir)
    code = generate_code(task_content)
    commit_and_push(repo_dir, code)
    # Get last commit hash
    repo = Repo(repo_dir)
    last_commit = next(repo.iter_commits())
    # Save to Astra
    save_to_astra(
        request=task_content,
        solution=code,
        result="committed",  # or any result string you want
        commit=last_commit.hexsha
    )
    print("Task run, committed, and saved to Astra DB.")

def get_latest_commit(repo_dir):
    repo = Repo(repo_dir)
    latest_commit = next(repo.iter_commits())
    print(f"Latest Commit: {latest_commit.hexsha}\nMessage: {latest_commit.message}\nDate: {latest_commit.committed_datetime}\n")

def get_embedding(text):
    # Configure the client with API key
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Get embedding using the embedding service
    result = genai.embed_content(
        model="models/embedding-001",
        content=text
    )
    
    # Return the embedding
    if hasattr(result, 'embedding'):
        return result.embedding
    elif isinstance(result, dict) and 'embedding' in result:
        return result['embedding']
    else:
        print(f"Unexpected embedding response format: {type(result)}")
        print(f"Response content: {result}")
        # Return a basic embedding as fallback
        return [0.0] * 768  # Standard embedding dimension

intents=[
    {"intent": "run the task", "action": "run_and_save"},
    {"intent": "search commit", "action": "search_commit"},
    {"intent": "check task", "action": "is_task_complete"},
    {"intent": "task ran", "action": "task_already_ran"},
    {"intent": "last solution", "action": "get_last_solution"},
    {"intent": "latest commit", "action": "get_latest_commit"}
]

action_map = {
    "run_and_save": run_and_save,
    "search_commit": search_commit,
    "is_task_complete": is_task_complete,
    "task_already_ran": task_already_ran,
    "get_last_solution": get_last_solution,
    "get_latest_commit": get_latest_commit
}

# save intent to db
if "intents" not in db.list_collection_names():
    db.create_collection("intents")
collection = db.get_collection("intents")

# Only insert intents if they don't already exist
# Check if intents already exist before adding
try:
    # For AstraPy client, find() only takes the query parameter
    existing_intents = list(collection.find({"intent": {"$exists": True}}))
    existing_intent_texts = [i.get("intent") for i in existing_intents]
    
    for intent in intents:
        if intent["intent"] not in existing_intent_texts:
            embedding = get_embedding(intent["intent"])
            collection.insert_one({
                "intent": intent["intent"],
                "embedding": embedding,
                "action": intent["action"]
            })
except Exception as e:
    print(f"Error when checking for existing intents: {e}")
    # If there's an error, just insert all intents
    for intent in intents:
        try:
            embedding = get_embedding(intent["intent"])
            collection.insert_one({
                "intent": intent["intent"],
                "embedding": embedding,
                "action": intent["action"]
            })
        except Exception as e:
            print(f"Failed to insert intent '{intent['intent']}': {e}")

def get_user_intent(user_input):
    try:
        user_embedding = get_embedding(user_input)
        
        # Try different vector search methods that might be available in AstraPy
        try:
            # First attempt: using find_vector method if available
            if hasattr(collection, 'find_vector'):
                results = collection.find_vector(
                    vector=user_embedding,
                    vector_field="embedding",
                    limit=1
                )
            # Second attempt: using vector_find method if available
            elif hasattr(collection, 'vector_find'):
                results = collection.vector_find(
                    vector=user_embedding,
                    field="embedding",
                    limit=1
                )
            # Third attempt: using similarity_search if available
            elif hasattr(collection, 'similarity_search'):
                results = collection.similarity_search(
                    query_vector=user_embedding,
                    field="embedding",
                    limit=1
                )
            # Last attempt: using find with $vectorSearch operator if MongoDB-style queries are supported
            else:
                print("Vector search methods not found. Falling back to direct intent matching.")
                # Fallback to simple string matching
                results = []
                for intent_data in collection.find({"intent": {"$exists": True}}):
                    if user_input.lower() in intent_data.get("intent", "").lower():
                        results.append({"action": intent_data.get("action"), "similarity": 1.0})
                        break
                
            # Process results
            if results and len(results) > 0:
                # Check if results is a list of documents or a cursor
                first_result = results[0] if isinstance(results, list) else next(results, None)
                
                if first_result:
                    # Check if the result has a similarity score
                    if isinstance(first_result, dict) and 'similarity' in first_result and first_result['similarity'] > 0.8:
                        return first_result.get('action')
                    # If no similarity score but action is available, return it
                    elif isinstance(first_result, dict) and 'action' in first_result:
                        return first_result.get('action')
                    # If it's a document with intent and action fields
                    elif hasattr(first_result, 'get') and first_result.get('action'):
                        return first_result.get('action')
        
        except Exception as e:
            print(f"Vector search failed: {e}")
            print("Falling back to simple string matching.")
            
        # Fallback to direct intent matching if vector search fails
        for intent in intents:
            if intent["intent"].lower() in user_input.lower():
                return intent["action"]
                
        return None
        
    except Exception as e:
        print(f"Error in get_user_intent: {e}")
        return None

def interactive_agent():
    repo_dir = "/home/system/projects/github-agent/agent-repo"
    clone_repo(repo_dir, REPO_URL)
    print("Agent initialized. Type 'exit' to quit.")
    while True:
        cmd = input("agent> ").strip()
        if cmd == "exit":
            break
        
        user_intent = get_user_intent(cmd)
        if user_intent:
            action_function = action_map.get(user_intent)
            if action_function:
                if user_intent == "search_commit":
                    keyword = input("Enter keyword to search for: ")
                    action_function(repo_dir, keyword)
                elif user_intent == "task_already_ran":
                    task_context = input("Enter task context: ")
                    action_function(repo_dir, task_context)
                else:
                    action_function(repo_dir)
            else:
                print(f"Action '{user_intent}' not found in action map")
        else:
            print("I didn't understand that command. Try: 'run the task', 'search commit', 'check task', 'task ran', 'last solution', 'latest commit', or 'exit'")

if __name__ == "__main__":
    interactive_agent()