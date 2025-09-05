from typing import List, Dict, Any

SAMPLES: List[str] = [
    "hey can you fix the login? users get 500 after oauth callback. pretty urgent. tag: auth, bug. maybe @sara can take it. need it done by Friday.",
    "minor css glitch on the mobile navbar; low priority. label=ui",
    "kafka consumer falls behind ~3h during nightly batch; escalate. assign to data team. deadline sept 20.",
    "please add dark mode sometime. not urgent",
    "prod outage in EU region; paging on-call; fix ASAP; labels: infra, incident, sev1",
]

# Gold annotations for evaluation (best-effort; edit to your needs)
GOLD: List[Dict[str, Any]] = [
    {"priority": "urgent", "owner": "sara", "tags": ["auth","bug"], "deadline": None},
    {"priority": "low", "owner": None, "tags": ["ui"], "deadline": None},
    {"priority": "high", "owner": None, "tags": ["data","kafka"], "deadline": None},  # date parsing can vary
    {"priority": "low", "owner": None, "tags": ["feature"], "deadline": None},
    {"priority": "urgent", "owner": None, "tags": ["infra","incident"], "deadline": None},
]
