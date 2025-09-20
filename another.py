# Install required packages (run only once)
# !pip install langchain langchain_core langchain_community langgraph langchain_openai

from langchain_openai.chat_models import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate

# Define the candidate state
class State(TypedDict):
    application: str
    experience_level: str
    skill_match: str
    response: str

# Initialize LLM
llm = ChatOpenAI()

# Initialize workflow
workflow = StateGraph(State)

# Node functions
def categorize_experience(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Based on the following job application, categorize the candidate as 'Entry-level', 'Mid-level' or 'Senior-level': {application}"
    )
    chain = prompt | llm
    experience_level = chain.invoke({"application": state["application"]}).content
    return {"experience_level": experience_level}

def assess_skillset(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Based on the job application, assess the candidate's skillset. Respond with either 'Match' or 'No Match': {application}"
    )
    chain = prompt | llm
    skill_match = chain.invoke({"application": state["application"]}).content
    return {"skill_match": skill_match}

def schedule_hr_interview(state: State) -> State:
    return {"response": "Candidate has been shortlisted for an HR interview."}

def escalate_to_recruiter(state: State) -> State:
    return {"response": "Candidate has senior-level experience but doesn't match job skills."}

def reject_application(state: State) -> State:
    return {"response": "Candidate doesn't meet JD and has been rejected."}

# Add nodes to workflow
workflow.add_node("categorize_experience", categorize_experience)
workflow.add_node("assess_skillset", assess_skillset)
workflow.add_node("schedule_hr_interview", schedule_hr_interview)
workflow.add_node("escalate_to_recruiter", escalate_to_recruiter)
workflow.add_node("reject_application", reject_application)

# Routing logic
def route_app(state: State) -> str:
    if state["skill_match"] == "Match":
        return "schedule_hr_interview"
    elif state["experience_level"] == "Senior-level":
        return "escalate_to_recruiter"
    else:
        return "reject_application"

# Add edges
workflow.add_edge(START, "categorize_experience")
workflow.add_edge("categorize_experience", "assess_skillset")
workflow.add_conditional_edges("assess_skillset", route_app)
workflow.add_edge("assess_skillset", END)
workflow.add_edge("schedule_hr_interview", END)
workflow.add_edge("escalate_to_recruiter", END)
workflow.add_edge("reject_application", END)

# Compile workflow
app = workflow.compile()

# Function to run candidate screening
def run_candidate_screening(application: str):
    results = app.invoke({"application": application})
    return {
        "experience_level": results["experience_level"],
        "skill_match": results["skill_match"],
        "response": results["response"]
    }

# Example usage
applications = [
    "I have 10 years of experience in software engineering with expertise in JAVA",
    "I have 1 year of experience in software engineering with expertise in JAVA",
    "I have experience in software engineering with expertise in Python",
    "I have 5 years of experience in software engineering with expertise in C++"
]

for app_text in applications:
    results = run_candidate_screening(app_text)
    print("\nApplication:", app_text)
    print("Experience Level:", results["experience_level"])
    print("Skill Match:", results["skill_match"])
    print("Response:", results["response"])
