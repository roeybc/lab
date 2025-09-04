import os, json
from typing import Tuple, Any
from openai import OpenAI
from jsonschema import Draft202012Validator
from dotenv import load_dotenv
load_dotenv()

MOVIE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "minLength": 1},
        "director": {"type": "string", "minLength": 3},
        "year": {"type": "integer", "minimum": 1888, "maximum": 2100},
        "rating": {"type": "string", "enum": ["G", "PG", "PG-13", "R", "NC-17"]},
        "release": {
            "type": "object",
            "properties": {
                "country_code": {
                    "type": "string",
                    "pattern": "^[A-Z]{2}$"  # exactly two-letter ISO code
                },
                "date": {
                    "type": "string",
                    "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$"  # strict YYYY-MM-DD
                }
            },
            "required": ["country_code", "date"],
            "additionalProperties": False
        },
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "actor": {"type": "string"}
                },
                "required": ["name", "actor"],
                "additionalProperties": False
            },
            "minItems": 5,  # must list at least 5 characters
            "maxItems": 10
        }
    },
    "required": ["title", "director", "year", "rating", "release", "characters"],
    "additionalProperties": False
}

VALIDATOR = Draft202012Validator(MOVIE_SCHEMA)
client = OpenAI()

PROMPTS = [
    'Return a movie JSON for "Titanic" with rating and a couple of main characters.',
    'JSON for "The Dark Knight" including rating and release info (US release date).',
    'Create a movie object for "Amélie" with characters and rating.',
    'Make a structured movie for "Blade Runner (Final Cut)" with 2–3 characters and release date.',
    'Give me a structured movie for "Spirited Away" (Japan release), include rating.',
    'JSON for Avatar with release country: US, release date: December 18 2009',
    'The Godfather'
]

def validate_json(data: Any) -> Tuple[bool, str]:
    VALIDATOR.validate(data)
    # try:
    #     VALIDATOR.validate(data)
    #     return True, ""
    # except Exception as e:
    #     return False, str(e)

def function_calling_call(q: str) -> Tuple[bool, Any, str]:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Return a JSON object that strictly matches the provided parameters schema."},
            {"role": "user",   "content": q}
        ],
        functions=[{
            "name": "return_movie",
            "description": "Return a structured Movie object",
            "parameters": MOVIE_SCHEMA
        }],
        function_call={"name": "return_movie"}
    )
    msg = resp.choices[0].message
    try:
        args = msg.function_call.arguments if msg.function_call else "{}"
        data = json.loads(args)
    except Exception as e:
        return False, None, f"arguments json error: {e}"

    # Validate against our schema
    ok, why = validate_json(data)
    return ok, data, why

def main(prompts: list[str]):
    successful = 0
    failed = 0
    
    for i, q in enumerate(prompts, 1):
        ok, data, err = function_calling_call(q)
        
        if ok:
            successful += 1
        else:
            failed += 1
        
        print(f"\n=== PROMPT {i}/{len(prompts)} ===")
        print(q)
        print(f"OK={ok}")
        
        if not ok:
            print("ERROR:", err)
            if isinstance(data, dict):
                preview = {k: data.get(k) for k in list(data.keys())[:5]}
                print("PREVIEW JSON:", json.dumps(preview, ensure_ascii=False))
    
    # Simple summary
    total = successful + failed
    success_rate = (successful / total * 100) if total > 0 else 0
    print(f"\nSUMMARY: {successful}/{total} successful ({success_rate:.1f}%)")
    print("Done.")

if __name__ == "__main__":
    main(prompts=PROMPTS)