from typing import List, Sequence

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph

from chains import generate_chain, reflect_chain

REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({"messages": state})


def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflect_chain.invoke({"messages": messages})
    return [HumanMessage(content=res.content)]


builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()

if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = HumanMessage(content="""Hi, I’m [NAME]. I am a Backend and Cloud Data Engineer with 15+ years of experience 
                          working primarily with Java, Python, and cloud platforms like AWS and GCP. 
                          For the past two years, I've been focused on Python and cloud technologies, 
                          where I worked on a Customer 360 project that involved processing real-time customer 
                          and order data from an e-commerce website to support analytics and decision-making. 
                          More recently, I’ve been involved in developing an AI product for image and 
                          video advertisements using GCP, Python, and Terraform. I’m passionate about AI, 
                          and AI agent-based work, where I can contribute to cloud infrastructure, orchestration, 
                          and automation.""")
    response = graph.invoke(inputs)
    print(response)
