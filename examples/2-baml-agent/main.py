import asyncio
from baml_client.async_client import b
from baml_client.types import Resume
from dotenv import load_dotenv
load_dotenv()


async def parse_resume(raw_resume: str) -> Resume: 
  # BAML's internal parser guarantees ExtractResume
  # to be always return a Resume type
  response = await b.ExtractResume(raw_resume)
  return response

if __name__ == "__main__":
    raw_resume = "I'm John and I'm an office worker at Dunder Mifflin, my email is john@dundermifflin.com, I know to sort papers, and love cakes"
    asyncio.run(parse_resume(raw_resume))
    