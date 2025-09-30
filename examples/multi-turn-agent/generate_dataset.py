"""
Generate synthetic multi-turn agent dataset with search and report tools.
Uses OpenAI API to generate realistic agent conversations.
"""

import json
import os
from typing import List, Dict
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the tools available to the agent
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for information on a given topic. Returns relevant search results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to execute"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": "Generate a comprehensive report based on gathered information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The title of the report"
                    },
                    "content": {
                        "type": "string",
                        "description": "The main content/findings to include in the report"
                    }
                },
                "required": ["title", "content"]
            }
        }
    }
]

# Topics for generating diverse examples
TOPICS = [
    "Compare the environmental impact of electric vehicles vs hydrogen fuel cell vehicles",
    "Analyze the benefits and drawbacks of remote work in tech companies",
    "Research the latest developments in quantum computing applications",
    "Investigate the impact of artificial intelligence on healthcare diagnostics",
    "Study the effectiveness of different renewable energy sources",
    "Examine cybersecurity threats in cloud computing",
    "Explore the relationship between diet and mental health",
    "Analyze trends in sustainable fashion and textile innovation",
    "Research the impact of social media on political discourse",
    "Compare different approaches to carbon capture technology",
    "Investigate the future of autonomous vehicles and transportation",
    "Study the effects of microplastics on marine ecosystems",
    "Analyze the economic impact of automation on manufacturing",
    "Research developments in gene therapy and personalized medicine",
    "Examine the role of blockchain in supply chain management",
    "Study the impact of urban planning on public health",
    "Analyze the effectiveness of different teaching methods in online education",
    "Research the psychology of consumer behavior in e-commerce",
    "Investigate the impact of climate change on agriculture",
    "Explore innovations in battery technology for energy storage"
]

def generate_search_results(query: str) -> str:
    """Simulate search results for a given query."""
    prompt = f"""Generate realistic search results for the query: "{query}"

Provide 3-4 key findings with specific facts, statistics, or insights.
Format as a natural search results summary.
Keep it concise but informative (3-5 sentences)."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )

    return response.choices[0].message.content.strip()


def generate_agent_conversation(topic: str) -> Dict:
    """Generate a complete multi-turn agent conversation for a given topic."""

    system_prompt = """You are a helpful research agent. You have access to two tools:
1. search(query) - to search for information
2. generate_report(title, content) - to create a final report

When given a research task, you should:
1. Break it down and search for relevant information (usually 2-3 searches)
2. Synthesize the findings
3. Generate a comprehensive report

Respond with tool calls in the proper format."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Please research and create a report on: {topic}"}
    ]

    conversation_turns = []
    max_turns = 6

    for turn in range(max_turns):
        # Get agent's response with tool calls
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=500
        )

        assistant_message = response.choices[0].message

        # Store the assistant's turn
        turn_data = {
            "role": "assistant",
            "content": assistant_message.content or ""
        }

        if assistant_message.tool_calls:
            turn_data["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in assistant_message.tool_calls
            ]

        conversation_turns.append(turn_data)
        messages.append(assistant_message)

        # If no tool calls, agent is done
        if not assistant_message.tool_calls:
            break

        # Execute tool calls and add results
        tool_messages = []
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            # Simulate tool execution
            if function_name == "search":
                result = generate_search_results(function_args["query"])
            elif function_name == "generate_report":
                result = f"Report '{function_args['title']}' generated successfully."
                # After report generation, we're done
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                }
                tool_messages.append(tool_msg)
                messages.append(tool_msg)
                conversation_turns.append(tool_msg)
                # Break after report generation
                return {
                    "topic": topic,
                    "messages": messages[1:],  # Exclude system message for training
                    "conversation_turns": conversation_turns
                }
            else:
                result = "Function not found."

            tool_msg = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            }
            tool_messages.append(tool_msg)
            messages.append(tool_msg)
            conversation_turns.append(tool_msg)

    return {
        "topic": topic,
        "messages": messages[1:],  # Exclude system message for training
        "conversation_turns": conversation_turns
    }


def generate_dataset(num_examples: int, output_file: str):
    """Generate the full dataset."""

    print(f"Generating {num_examples} examples...")
    dataset = []

    for i in range(num_examples):
        topic = TOPICS[i % len(TOPICS)]
        print(f"Generating example {i+1}/{num_examples}: {topic[:60]}...")

        try:
            conversation = generate_agent_conversation(topic)
            dataset.append(conversation)
        except Exception as e:
            print(f"Error generating example {i+1}: {e}")
            continue

    # Save dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"\nDataset saved to {output_file}")
    print(f"Total examples: {len(dataset)}")

    # Print statistics
    total_turns = sum(len(ex['conversation_turns']) for ex in dataset)
    avg_turns = total_turns / len(dataset) if dataset else 0
    print(f"Average turns per conversation: {avg_turns:.2f}")


if __name__ == "__main__":
    output_file = "data/multi_turn_agent_dataset.json"
    num_examples = 50  # Generate 50 examples

    generate_dataset(num_examples, output_file)