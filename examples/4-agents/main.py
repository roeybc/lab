"""
Simple ReAct (Reasoning and Acting) Agent Implementation

This demonstrates a ReAct loop using OpenAI's structured output and function calling:
1. Agent receives user query
2. Uses function calling to execute tools when needed
3. Returns structured responses
4. Continues until task is complete

The agent uses proper OpenAI function calling instead of prompt parsing.
"""

import asyncio
import json
import math
import os
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()
client = AsyncOpenAI()

class ActionType(Enum):
    """Enum for different action types the agent can take"""
    TOOL_CALL = "tool_call"
    FINAL_ANSWER = "final_answer"
    CONTINUE = "continue"

# Pydantic models for structured output
class ToolCall(BaseModel):
    """Represents a tool call to be executed"""
    id: str
    function_name: str
    arguments: Dict[str, Any]

class ReasoningResult(BaseModel):
    """Result from the reasoning phase"""
    action_type: ActionType
    content: Optional[str] = None
    tool_calls: List[ToolCall] = []
    raw_message: Optional[str] = None

class Tools:
    """Available tools for the agent to use via function calling"""
    
    @staticmethod
    def calculator(expression: str) -> str:
        """Safely evaluate mathematical expressions"""
        try:
            # Only allow safe mathematical operations
            allowed_names = {
                k: v for k, v in math.__dict__.items() 
                if not k.startswith("__")
            }
            allowed_names.update({"abs": abs, "round": round, "pow": pow})
            
            # Basic safety check - remove potentially dangerous characters
            if any(char in expression for char in ['import', '__', 'exec', 'eval', 'open', 'file']):
                return "Error: Potentially unsafe expression"
            
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"Calculation result: {result}"
        except Exception as e:
            return f"Calculation error: {str(e)}"
    
    @staticmethod
    def search(query: str) -> str:
        """Search for information (mock implementation)"""
        mock_results = {
            "python": "Python is a high-level programming language known for its simplicity and readability.",
            "react": "ReAct (Reasoning and Acting) is a paradigm that combines reasoning and acting in language models.",
            "openai": "OpenAI is an AI research company that created GPT models and ChatGPT.",
            "weather": "Today's weather is sunny with a temperature of 72°F.",
            "javascript": "JavaScript is a programming language commonly used for web development.",
            "ai": "Artificial Intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence.",
        }
        
        query_lower = query.lower()
        for key, value in mock_results.items():
            if key in query_lower:
                return f"Search result: {value}"
        
        return f"Search result: No specific information found for '{query}'. This is a mock search implementation."

class Agent:
    """ReAct Agent using OpenAI's structured output and function calling"""
    
    def __init__(self):
        self.tools = Tools()
        self.max_iterations = 10
        
        # Define function schemas for OpenAI function calling
        self.function_definitions = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate mathematical expressions safely",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Mathematical expression to evaluate (e.g., '2+2', '15*23+47')"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "search",
                    "description": "Search for information on a topic",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query to find information about"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    
    async def reason(self, messages: List[Dict[str, str]], use_functions: bool = True) -> ReasoningResult:
        """
        Core reasoning method - calls OpenAI and determines next action
        
        Args:
            messages: Conversation messages
            use_functions: Whether to enable function calling
            
        Returns:
            ReasoningResult with action type and any tool calls to execute
        """
        try:
            kwargs = {
                "model": "gpt-4o-mini",
                "messages": messages,
                "temperature": 0.1
            }
            
            if use_functions:
                kwargs["tools"] = self.function_definitions
                kwargs["tool_choice"] = "auto"
            
            response = await client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            
            # Determine action type based on response
            if message.tool_calls:
                # Agent wants to use tools
                tool_calls = []
                for tool_call in message.tool_calls:
                    tool_calls.append(ToolCall(
                        id=tool_call.id,
                        function_name=tool_call.function.name,
                        arguments=json.loads(tool_call.function.arguments)
                    ))
                
                return ReasoningResult(
                    action_type=ActionType.TOOL_CALL,
                    content=message.content,
                    tool_calls=tool_calls,
                    raw_message=message.content
                )
            
            elif message.content:
                return ReasoningResult(
                    action_type=ActionType.FINAL_ANSWER,
                    content=message.content,
                    raw_message=message.content
                )

            else:
                # No content, continue
                return ReasoningResult(
                    action_type=ActionType.CONTINUE,
                    content="No response from model",
                    raw_message=""
                )
                
        except Exception as e:
            return ReasoningResult(
                action_type=ActionType.CONTINUE,
                content=f"Error in reason method: {str(e)}",
                raw_message=""
            )
    
    def add_tool_calls_to_memory(self, messages: List[Dict[str, Any]], reasoning_result: ReasoningResult) -> None:
        """
        Add tool calls to memory (messages list)
        
        Args:
            messages: The conversation messages list to append to
            reasoning_result: The reasoning result containing tool calls to add
        """
        # Add assistant message with tool calls
        messages.append({
            "role": "assistant",
            "content": reasoning_result.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function_name,
                        "arguments": json.dumps(tool_call.arguments)
                    }
                } for tool_call in reasoning_result.tool_calls
            ]
        })
    
    def add_tool_result_to_memory(self, messages: List[Dict[str, Any]], tool_call_id: str, result: str) -> None:
        """
        Add tool result to memory (messages list)
        
        Args:
            messages: The conversation messages list to append to
            tool_call_id: The ID of the tool call this result corresponds to
            result: The result content from the tool execution
        """
        # Add tool result to conversation
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result
        })
    
    def act(self, function_name: str, arguments: Dict[str, Any]) -> str:
        """
        Acting method - executes tool functions
        
        Args:
            function_name: Name of the function to execute
            arguments: Arguments to pass to the function
            
        Returns:
            String result from the function execution
        """
        try:
            if hasattr(self.tools, function_name):
                function = getattr(self.tools, function_name)
                return function(**arguments)
            else:
                return f"Error: Unknown function '{function_name}'"
        except Exception as e:
            return f"Error executing {function_name}: {str(e)}"
    
    async def react_loop(self, user_query: str) -> str:
        """
        Main ReAct loop with clear separation of reasoning and acting
        """
        print(f"\n🎯 Observe: {user_query}")
        
        messages = [
            {
                "role": "system", 
                "content": """You are a helpful AI agent. Use the available functions when you need to perform calculations or search for information. 

Think step by step and use tools when needed. When you have enough information to answer the user's question completely, provide a final answer."""
            },
            {"role": "user", "content": user_query}
        ]
        
        for iteration in range(self.max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # REASON: Get next action from the model
            reasoning_result = await self.reason(messages)
            print(f"🧠 Reasoning: {reasoning_result.action_type.value}")
            
            if reasoning_result.action_type == ActionType.TOOL_CALL:
                print("🔧 Act:")
                
                # Add assistant message with tool calls to memory
                self.add_tool_calls_to_memory(messages, reasoning_result)
                
                # ACT: Execute each tool call
                for tool_call in reasoning_result.tool_calls:
                    print(f"  📱 {tool_call.function_name}({tool_call.arguments})")
                    
                    # Execute the tool
                    result = self.act(tool_call.function_name, tool_call.arguments)
                    print(f"  📋 Result: {result}")
                    
                    # Add tool result to memory
                    self.add_tool_result_to_memory(messages, tool_call.id, result)
                
                # Continue the ReAct loop
                continue
                
            elif reasoning_result.action_type == ActionType.FINAL_ANSWER:
                print(f"✅ Final Answer: {reasoning_result.content}")
                return reasoning_result.content
                
            elif reasoning_result.action_type == ActionType.CONTINUE:
                print(f"💭 Agent response: {reasoning_result.content}")
                
                # Add response and ask for final answer
                messages.append({"role": "assistant", "content": reasoning_result.content})
                messages.append({"role": "user", "content": "Please provide your final answer to the original question."})
                continue
        
        return "Maximum iterations reached. Unable to complete the task."

async def demo():
    """Demonstrate the ReAct agent using structured output and function calling"""
    agent = Agent()
    
    examples = [
        "What is 15 * 23 + 47?",
        "I need to know about Python programming and then calculate 100 divided by 4",
        "Search for information about ReAct and tell me what it is",
        "Calculate the square root of 144 and then search for information about OpenAI",
    ]
    
    for i, query in enumerate(examples, 1):
        print(f"\n{'='*70}")
        print(f"EXAMPLE {i}: {query}")
        print(f"{'='*70}")
        
        result = await agent.react_loop(query)
        print(f"\n🎯 FINAL RESULT: {result}")
        print("\n" + "="*70)

async def interactive_demo():
    """Interactive demo where user can ask questions"""
    agent = Agent()
    
    print("🤖 ReAct Agent Interactive Demo")
    print("Using OpenAI function calling and structured output")
    print("Available tools: calculator, search")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("🧑 Your question: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
                
            result = await agent.react_loop(user_input)
            print(f"\n🤖 Agent: {result}\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        asyncio.run(interactive_demo())
    else:
        asyncio.run(demo())
