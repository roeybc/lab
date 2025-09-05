from openai import OpenAI
import instructor
from schema import Task
from prompts import SYSTEM, USER_TEMPLATE
from dotenv import load_dotenv
load_dotenv()

client = instructor.from_openai(OpenAI())

def extract_task(text: str, max_retries: int = 2) -> Task:
    return client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=Task,
        temperature=0,
        messages=[
            {"role":"system","content": SYSTEM},
            {"role":"user","content": USER_TEMPLATE.format(TEXT=text)},
        ],
        max_retries=max_retries,
    )

def main():
    demo = "draft the demo slides for Friday; urgent; assign to @alex; tag: presentation"
    print(extract_task(demo).model_dump_json(indent=2))

if __name__ == "__main__":
    main()
