from baml_client import b
from baml_client.types import Task
from dotenv import load_dotenv
load_dotenv()


def extract_task(text: str) -> Task:
    task = b.ExtractTask(text=text)
    return task

def main():
    demo = "draft the demo slides for Friday; urgent; assign to @alex; tag: presentation"
    print(extract_task(demo).model_dump_json(indent=2))

if __name__ == "__main__":
    main()
