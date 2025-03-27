'''import os
from dotenv import load_dotenv
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from duckduckgo_search import DDGS
from openai import OpenAI


load_dotenv()


api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API Key is not loading")

client = OpenAI(api_key=api_key)

app = FastAPI()

class CourseRequest(BaseModel):
    description: str

def research_content(description: str):
    with DDGS() as ddgs:
        search_results = [r["href"] for r in ddgs.text(description, max_results=5)]
    return search_results

def generate_modules(description: str):
    prompt = f"Generate 5-6 structured module titles for a course on: {description}."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}]
    )
    return response.choices[0].message.content.split("\n")

def generate_content(modules):
    contents = []
    for module in modules:
        prompt = f"Write detailed educational content for the course module: {module}."
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}]
        )
        contents.append(response.choices[0].message.content)
    return contents

@app.post("/generate_course/")
async def generate_course(request: CourseRequest):
    research = research_content(request.description)
    modules = generate_modules(request.description)
    content = generate_content(modules)

    structured_course = {"description": request.description, "modules": []}
    for i, module in enumerate(modules):
        structured_course["modules"].append({
            "title": module,
            "content": content[i]
        })

    return structured_course

'''
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)

app = FastAPI()


class CourseRequest(BaseModel):
    description: str



search_tool = DuckDuckGoSearchRun()


def research_agent(state):
    search_results = search_tool.run(state["description"])
    return {**state, "research_results": search_results}

def structuring_agent(state):
    prompt = PromptTemplate.from_template("""
    Given the following research results:
    {research_results}
    Generate a structured outline with 5-6 modules for a course titled: {description}
    """)
    modules = llm.predict(prompt.format(research_results=state["research_results"], description=state["description"]))
    return {**state, "modules": modules.split("\n")}

def content_generation_agent(state):
    content = {}
    for module in state["modules"]:
        prompt = PromptTemplate.from_template("""
        Write a detailed lesson for the module: {module} in a structured and educational format.
        Include key concepts, explanations, examples, and interactive elements.
        """)
        content[module] = llm.predict(prompt.format(module=module))
    return {**state, "content": content}


course_graph = StateGraph(dict)
course_graph.add_node("research", research_agent)
course_graph.add_node("structure", structuring_agent)
course_graph.add_node("generate_content", content_generation_agent)

course_graph.add_edge("research", "structure")
course_graph.add_edge("structure", "generate_content")

course_graph.set_entry_point("research")
workflow = course_graph.compile()

# API Endpoint
@app.post("/generate_course/")
def generate_course(request: CourseRequest):
    initial_state = {"description": request.description, "research_results": [], "modules": [], "content": {}}
    final_state = workflow.invoke(initial_state)
    return {"modules": final_state["modules"], "content": final_state["content"]}
