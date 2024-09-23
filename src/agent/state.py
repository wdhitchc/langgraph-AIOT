"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
import operator
from typing import Any
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict



@dataclass(kw_only=True)
class InputState:
    """Defines the input state for the agent, representing a narrower interface to the outside world. in our case just the user input"""
    user_input: str


@dataclass(kw_only=True)
class State(InputState):
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    for more information.
    """


    messages: Annotated[list[AnyMessage], add_messages]
    is_final: bool = False
    sender: Annotated[list[str], operator.concat]
    iteration: int = field(default=1)
    max_iterations: int = 10


    def format_prompt_history_for_prompt(self):
        prompt_history = f"**Important** Initial Query: {self.user_input}\n\n"

        # Adjust the loop to handle indexing correctly
        for i in range(0, len(self.messages), 2):  # Step through the messages in pairs
            # Check if there is a corresponding LLM message after the agent's message
            agent_message = self.messages[i].content
            # Append the current iteration's messages to the prompt history
            prompt_history += f"**Iteration** {i//2 + 1} / {self.max_iterations}"
            prompt_history += f"Cognitive Reflection Agent: {agent_message}\n\n\n"
            if i + 1 < len(self.messages):
             llm_message = self.messages[i + 1].content
             prompt_history += f"LLM answer: {llm_message}\n\n\n"
        
        return prompt_history
    
@dataclass(kw_only=True)
class OutputState(State):
    """Defines the output state for the agent, representing a narrower interface to the outside world.
    This class is used to define the structure of outgoing data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    for more information.
    """
    answer: str = "I was unable to find an anser due to an error"