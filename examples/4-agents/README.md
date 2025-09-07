# ReAct Agent with Structured Output & Function Calling

This example demonstrates a **really simple** ReAct (Reasoning and Acting) agent using OpenAI's structured output and function calling APIs.

## What is ReAct?

ReAct is a paradigm that combines **Reasoning** and **Acting** in language models. Instead of parsing text responses, this implementation uses:

- **OpenAI Function Calling** - Structured tool execution
- **Pydantic Models** - Type-safe data structures  
- **Agent Class** - Clean abstraction with `ask()` method

## Architecture

### Agent Class
```python
class Agent:
    async def ask(messages, use_functions=True) -> Dict
    async def react_loop(user_query) -> str
```

### Key Features
- **🔧 Function Calling**: Native OpenAI tool integration
- **📝 Structured Output**: Pydantic models for type safety
- **🎯 Clean API**: Simple `ask()` method wraps all OpenAI calls
- **🔄 ReAct Loop**: Automatic reasoning and acting cycle

## Available Tools

- `calculator(expression)` - Safe mathematical evaluation
- `search(query)` - Information search (mock implementation)

## Usage

```bash
# Install dependencies
uv sync

# Run demo examples
python main.py

# Interactive mode
python main.py interactive
```

## Example: Function Calling in Action

**User**: "What is 15 * 23 + 47?"

**Agent Process**:
```
🎯 User Query: What is 15 * 23 + 47?

--- Iteration 1 ---
🔧 Using tools:
  📱 calculator({'expression': '15 * 23 + 47'})
  📋 Result: Calculation result: 392

--- Iteration 2 ---
💭 Agent response: The calculation 15 * 23 + 47 equals 392.
✅ Final Answer: The calculation 15 * 23 + 47 equals 392.
```

**Behind the scenes**:
1. Agent receives structured function definition for `calculator`
2. OpenAI decides to call `calculator` function with proper arguments
3. Function executes and returns structured result
4. Agent provides final answer

## Multi-step Example

**User**: "Search for Python info and calculate 100/4"

**Process**:
```
--- Iteration 1 ---
🔧 Using tools:
  📱 search({'query': 'Python programming'})
  📋 Result: Search result: Python is a high-level programming language...

--- Iteration 2 ---  
🔧 Using tools:
  📱 calculator({'expression': '100/4'})
  📋 Result: Calculation result: 25.0

--- Iteration 3 ---
✅ Final Answer: Python is a high-level programming language known for its simplicity and readability. The calculation 100 divided by 4 equals 25.0.
```

## Adding New Tools

1. Add method to `Tools` class:
```python
@staticmethod
def new_tool(param: str) -> str:
    return "Tool result"
```

2. Add function definition to `Agent.__init__()`:
```python
{
    "type": "function",
    "function": {
        "name": "new_tool",
        "description": "What the tool does",
        "parameters": {
            "type": "object", 
            "properties": {
                "param": {"type": "string", "description": "Parameter description"}
            },
            "required": ["param"]
        }
    }
}
```

## Key Advantages

- **🚀 No Text Parsing**: Uses native OpenAI function calling
- **🔒 Type Safety**: Pydantic models ensure data integrity  
- **🎯 Reliable**: Structured I/O eliminates parsing errors
- **📈 Scalable**: Easy to add complex tools and workflows
- **🧹 Clean Code**: Separation of concerns with Agent class

## Requirements

- Python 3.11+
- OpenAI API key (`.env` file: `OPENAI_API_KEY=your_key`)
- Dependencies: `openai`, `python-dotenv`, `pydantic`

This implementation showcases modern best practices for building ReAct agents with structured APIs.
