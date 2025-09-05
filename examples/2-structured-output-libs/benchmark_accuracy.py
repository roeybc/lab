import asyncio
from typing import Dict, Any, List
from rich.table import Table
from rich.console import Console
from schema import Task
from data import SAMPLES, GOLD
from eval import score_sample, aggregate
from dotenv import load_dotenv
load_dotenv()


console = Console()

def run_runner(label: str, mod, fn_name="extract_task"):
    fn = getattr(mod, fn_name)
    preds: List[Dict[str, Any]] = []
    scores = []
    for i, text in enumerate(SAMPLES):
        try:
            # Handle both sync and async functions
            if asyncio.iscoroutinefunction(fn):
                task: Task = asyncio.run(fn(text))
            else:
                task: Task = fn(text)
            preds.append(task.model_dump())
        except Exception as e:
            preds.append({})  # treat as empty prediction
        scores.append(score_sample(preds[-1], GOLD[i]))
    acc = aggregate(scores)

    table = Table(title=f"{label} — accuracy")
    table.add_column("Metric"); table.add_column("Score")
    for k,v in acc.items():
        table.add_row(k, f"{v:.3f}")
    console.print(table)
    return acc

def main():
    results = {}

    # Test Instructor
    try:
        import instructor_demo
        results["Instructor"] = run_runner("Instructor", instructor_demo)
    except Exception as e:
        console.print(f"[red]Failed to run Instructor: {e}[/red]")

    # Test PydanticAI
    try:
        import pydanticai_demo
        results["PydanticAI"] = run_runner("PydanticAI", pydanticai_demo)
    except Exception as e:
        console.print(f"[red]Failed to run PydanticAI: {e}[/red]")

    # Test BAML
    try:
        import baml_demo
        results["BAML"] = run_runner("BAML", baml_demo)
    except Exception as e:
        console.print(f"[red]Failed to run BAML: {e}[/red]")

    console.print("[bold]\nSummary[/bold]")
    for k, v in results.items():
        console.print(k, v)

if __name__ == "__main__":
    main()
