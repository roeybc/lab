import asyncio
from typing import List
from memory import ShortTermMemory, MemoryStore
from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI


class Agent:
    def __init__(self):
        self.stm = ShortTermMemory()
        self.ltm = MemoryStore()
        self.client = AsyncOpenAI()
    
    async def _answer_query(self, context: List[dict]) -> dict:
        """Generate an answer based on the provided context."""
        output = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=context
        )
        
        answer = output.choices[0].message.content
        return {"role": "assistant", "content": answer}
    
    def _build_context(self, query: str) -> List[dict]:
        """Build context from STM and LTM for the given query."""
        recent = self.stm.last_window()
        recalled = self.ltm.search(query)
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        
        for role, content in recent:
            messages.append({"role": role, "content": content})
        
        if recalled:
            messages.append({"role": "system", "content": "Relevant facts:\n" + "\n".join(recalled)})
        
        messages.append({"role": "user", "content": query})
        return messages
    
    async def ask(self, query: str) -> str:
        """
        Process a query using both short-term and long-term memory.
        
        Args:
            query: The user's question or input
            
        Returns:
            The agent's response as a string
        """
        # Add query to memory
        self.stm.append("user", query)
        await self.ltm.save(query)
        
        # Build context using both STM and LTM
        context = self._build_context(query)
        
        # Generate response
        response = await self._answer_query(context)
        
        # Add response to memory
        answer = response["content"]
        self.stm.append("assistant", answer)
        await self.ltm.save(answer)
        
        return answer


async def main():
    agent = Agent()
    
    # First interaction - provide some information
    input_text = "I live in NYC and love anchovy pizza."
    response1 = await agent.ask(input_text)
    print(f"User: {input_text}")
    print(f"Agent: {response1}")
    
    # Second interaction - query the information
    query = "Where do I live and what topping do I like?"
    response2 = await agent.ask(query)
    print(f"User: {query}")
    print(f"Agent: {response2}")


if __name__ == "__main__":
    asyncio.run(main())