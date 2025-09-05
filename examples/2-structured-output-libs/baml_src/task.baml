class Task {
  description string
  priority "low" | "medium" | "high" | "urgent"@description("priority")
  owner string?
  tags string[]@description("<= 8, lowercase, single tokens if possible")
  deadline string?@description("ISO-8601 (YYYY-MM-DD) if clearly stated, else null")
  confidence float@description("0..1 calibrated; 0.5 = unsure, 0.9 = very sure")
}

function ExtractTask(text: string) -> Task {
  client "openai/gpt-4o" // Set OPENAI_API_KEY to use this client.
  prompt #"
    Extract from this content:
    {{ text }}

    {{ ctx.output_format }}
  "#

}
