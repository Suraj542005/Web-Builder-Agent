from langchain_groq import ChatGroq
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from prompts import *
from states import *
from langgraph.constants import END
from langgraph.graph import StateGraph
from langchain.globals import set_debug, set_verbose
from tools import *
from langgraph.prebuilt import create_react_agent
import os

# Gives the Internal Details
set_debug(True)
set_verbose(True)

# Loading API Key's
load_dotenv()

# Selecting the LLM Model
# llm = ChatGroq(model="openai/gpt-oss-120b")

llm = ChatGoogleGenerativeAI(
    # model="gemini-2.5-pro",   # or "gemini-1.5-flash" if you want cheaper/faster
    model="gemini-2.5-flash",   # or "gemini-1.5-flash" if you want cheaper/faster
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

""" Custom Functions """


def enhanceUserPrompt(userPrompt: str) -> str:
    resp = llm.invoke(f"Please enhance this given prompt in clear and concise way with few lines {userPrompt}")
    return resp.content


def planner_agent(state: dict) -> dict:
    users_prompt = state["user_prompt"]
    resp = llm.with_structured_output(Plan).invoke(planner_prompt(users_prompt))
    if resp is None:
        raise ValueError("Planner did not return a Valid Response.")

    time.sleep(20)

    return {"plan": resp}


def architect_agent(state: dict) -> dict:
    plan: Plan = state["plan"]
    resp = llm.with_structured_output(TaskPlan).invoke(architect_prompt(plan))
    if resp is None:
        raise ValueError("Architect did not return a Valid Response.")
    resp.plan = plan

    time.sleep(20)

    return {"task_plan": resp}


def coder_agent(state: dict) -> dict:
    coder_state = state.get("coder_state")
    if coder_state is None:
        coder_state = CoderState(task_plan=state["task_plan"], current_step_idx=0)

    steps = coder_state.task_plan.implementation_steps
    if coder_state.current_step_idx >= len(steps):
        return {"coder_state": coder_state, "status": "DONE"}

    # current_step_idx = 0
    current_task = steps[coder_state.current_step_idx]

    existing_content = read_file.run(current_task.filepath)

    user_prompt = (
        f"Task: {current_task.task_description}\n"
        f"File: {current_task.filepath}\n"
        f"Existing content:\n{existing_content}\n"
        "Use write_file(path, content) to save your changes."
    )
    system_prompt = coder_system_prompt()
    coder_tools = [read_file, write_file, list_files, get_current_directory]
    react_agent = create_react_agent(llm, coder_tools)
    react_agent.invoke({
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}]
    })
    coder_state.current_step_idx += 1

    time.sleep(20)

    return {"coder_state": coder_state}


""" END """


# res = llm.invoke("What is Agentic AI, explain mi in hing-lish and in short, also give me some project ideas name")
# res = llm.invoke("Give me some leetcode questions related to topic DP, so i can improve my skills.")
# res = llm.with_structured_output(Plan).invoke(prompt)

# print(res.content)
# print(res)

graph = StateGraph(dict)
graph.add_node("planner", planner_agent)
graph.add_node("architect", architect_agent)
graph.add_node("coder", coder_agent)


graph.add_edge("planner", "architect")
graph.add_edge("architect", "coder")
graph.add_conditional_edges(
    "coder", lambda s: "END" if s.get("status") == "DONE" else "coder",
    {"END": END, "coder": "coder"}
)

graph.set_entry_point("planner")

agent = graph.compile()

if __name__ == "__main__":
    # user_prompt = "Create a Crazy looking and working Calculator"
    # user_prompt = "Create a Crazy looking game development websites which shows the crazy looking visuals"
    # user_prompt = "I want you to make a cool chat web application that "
    # user_prompt = enhanceUserPrompt(user_prompt)
    # print(user_prompt)
    user_prompt = """
        "I want to build the frontend of a Spotify clone website.
        tech stack: use only "HTML, CSS, and javaScript"
    """

    res = agent.invoke({"user_prompt": user_prompt}, {"recursion_limit": 100})
    print(res)
