import asyncio
import os, re, json, sys, time
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()
client = AsyncOpenAI()

SYSTEM = "Return ONLY a JSON object matching the requested schema."
SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "director": {"type": "string"},
        "year": {"type": "integer"}
    },
    "required": ["title","director","year"],
    "additionalProperties": False
}
SCHEMA_HINT = """Return ONLY JSON with:
{"title": string, "director": string, "year": number}
"""
PROMPTS = [
    'Inception (2010) by Christopher Nolan. Return the object AND a brief explanation, please',
    'Give me two options: "Seven Samurai" and "Ikiru"',
    'Output as YAML: title: Spirited Away, director: Hayao Miyazaki, year: 2001',
    'Title: "The Godfather", Year: "1972", Director: "Francis Ford Coppola"',
    '"Se7en" (1995) dir. David Fincher — include cast array too. Make sure to include a field "notes" with a short critique',
    'Ikiru — if unsure, say “not sure” in plain text after the JSON',
    'Titanic (James Cameron, year nineteen ninety seven)',
    'Please return the JSON object for "The Godfather (1972)" directed by Francis Ford Coppola.',
    'Create a JSON object for "Alien (Director’s Cut)". Don\'t display the year.',
    'Can you give me the JSON for "Interstellar"? It\'s one of my favorite sci-fi films. what do you think about it?',
    'Could you return a JSON object for the movie "Whiplash"?'
]

def try_extract_json(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m: return False, None
    try: return True, json.loads(m.group(0))
    except: return False, None

def valid_movie(d):
    if not isinstance(d, dict):
        return False
    
    title = d.get("title")
    director = d.get("director")
    year = d.get("year")
    
    # Basic type checks
    if not isinstance(title, str) or not isinstance(director, str):
        return False
    
    # Year validation - accept int or string that can be converted to int
    if isinstance(year, str):
        try:
            year_int = int(year)
        except ValueError:
            return False
    elif isinstance(year, int):
        year_int = year
    else:
        return False
    
    # Check if year starts with 19 or 2 (1900-1999 or 2000-2999)
    if not (1900 <= year_int <= 2027):
        return False
    
    # Check if title contains year in parentheses (potential issue)
    import re
    year_in_title = re.search(r'\((19\d{2}|2\d{3})\)', title)
    if year_in_title:
        title_year = int(year_in_title.group(1))
        # If the year in title doesn't match the year field, it's suspicious
        if title_year != year_int:
            return False
    
    # Check for no redundant fields - only allow title, director, year
    allowed_fields = {"title", "director", "year"}
    if set(d.keys()) != allowed_fields:
        return False
    
    return True

async def raw_json(prompt: str):
    output = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":SYSTEM},
                  {"role":"user","content":SCHEMA_HINT + f'\nCreate object for: "{prompt}"'}]
    )
    
    results = output.choices[0].message.content
    ok, data = try_extract_json(results)
    return ok and valid_movie(data), data, results

async def function_call(prompt: str):
    output = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":"Extract a movie object."},
                  {"role":"user","content":prompt}],
        functions=[{"name":"return_movie","description":"Movie object","parameters":SCHEMA}],
        function_call={"name":"return_movie"}
    )
    args = output.choices[0].message.function_call.arguments
    try:
        data = json.loads(args)
        return valid_movie(data), data, args
    except Exception as e:
        return False, {"error": str(e), "raw": args}, args

async def benchmark(prompts):
    results = {"raw_json":{"ok":0,"total":0},"function_call":{"ok":0,"total":0}}
    
    for q in prompts:
        q = q.strip()
        if not q: continue
        
        # Call raw_json function
        ok, data, raw = await raw_json(q)
        results["raw_json"]["ok"] += int(ok)
        results["raw_json"]["total"] += 1
        if not ok:
            print(f"\n[RAW FAILED] {q}\nDATA={data}\nRAW={raw[:180]}...\n")
        
        # Call function_call function
        ok, data, raw = await function_call(q)
        results["function_call"]["ok"] += int(ok)
        results["function_call"]["total"] += 1
        if not ok:
            print(f"\n[FUNC FAILED] {q}\nDATA={data}\nRAW={raw[:180]}...\n")
    
    for k in results:
        ok, tot = results[k]["ok"], results[k]["total"]
        rate = ok/tot if tot else 0.0
        print(f"{k}: {ok}/{tot} = {rate:.1%}")

if __name__ == "__main__":
    asyncio.run(benchmark(PROMPTS))
