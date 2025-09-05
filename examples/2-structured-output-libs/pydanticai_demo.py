import asyncio
from pydantic_ai import Agent
from schema import Task
from prompts import SYSTEM, USER_TEMPLATE
from dotenv import load_dotenv
load_dotenv()

agent = Agent(
    "openai:gpt-4o-mini",
    instructions=SYSTEM,
    output_type=Task,
)

async def extract_task(text: str) -> Task:
    result = await agent.run(USER_TEMPLATE.format(TEXT=text))
    return result.output

async def main():
    demo = "draft the demo slides for Friday; urgent; assign to @alex; tag: presentation"
    result = await extract_task(demo)
    print(result.model_dump_json(indent=2))

if __name__ == "__main__":
    asyncio.run(main())
