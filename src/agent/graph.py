from typing import Any, Dict, cast

from agent.utils import init_model
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from agent.prompts import llm_agent_prompt, ida_agent_prompt, get_ida_prompt_variables, get_llm_agent_prompt_variables
from agent.configuration import Configuration 
from agent.state import InputState, OutputState, State
from agent.node_response_models import BrainIteration, LLMResponse, GraphOutput
from langchain_core.messages import AIMessage, HumanMessage

max_iterations = 15


async def IDA_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    chat_model = init_model().with_structured_output(BrainIteration)
    runnable = ida_agent_prompt | chat_model
    response = cast(BrainIteration, await runnable.ainvoke(get_ida_prompt_variables(state)))
    return {
        "messages": [HumanMessage(content= response.self_thought)],
        "sender" : ["IDA_agent"],
        "is_final": response.iteration_stop,
        'iteration': state.iteration
    }

async def LLM_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    chat_model = init_model().with_structured_output(LLMResponse)
    runnable = llm_agent_prompt | chat_model
    response = cast(LLMResponse, await runnable.ainvoke(get_llm_agent_prompt_variables(state)))
    return {
        "messages": [AIMessage(content=response.response)],
        "sender" : ["LLM_agent"],
        "is_final": response.answer_to_query,
        "iteration": state.iteration + 1
    }

async def router(state: State, config: RunnableConfig) -> Dict[str, Any]:

    if state.is_final == True or state.iteration > max_iterations:
        # Any agent decided the work is done, or max iterations has been reached

        #this may be the appropriate way to do this in langgraph
        return "__end__"
    
    elif state.sender[-1] == "IDA_agent":
        return "LLM_agent"
    else:
        return "IDA_agent"


# Define a new graph
workflow = StateGraph(State, input=InputState, output=OutputState, config_schema=Configuration)

# Add the node to the graph
workflow.add_node("IDA_agent", IDA_agent)
workflow.add_node("LLM_agent", LLM_agent)


workflow.add_edge('__start__', 'IDA_agent')
workflow.add_conditional_edges("IDA_agent", router)
workflow.add_conditional_edges("LLM_agent", router)


# Compile the workflow into an executable graph
graph = workflow.compile()
graph.name = "Iteration of Thought"  # This defines the custom name in LangSmith

