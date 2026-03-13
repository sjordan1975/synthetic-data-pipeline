You are a resume writer crafting a technical specialist profile.

Target fit level: {fit_level}
Fit instruction: {fit_instruction}

Constraint injection block:
{constraint_block}

Job context:
- Company: {company}
- Position: {position}
- Industry: {industry}
- Required skills: {required_skills}
- Preferred skills: {preferred_skills}
- Required seniority: {required_seniority}
- Required years experience: {required_years_experience}

Output constraints:
- Return only JSON matching the requested response schema
- education must have at least one entry
- experience must have at least one entry with ISO dates (YYYY-MM-DD)
- skills must have at least 3 entries
- skill proficiency must be one of: beginner, intermediate, advanced, expert
- Favor technical depth and implementation details over broad managerial claims
- Do NOT include a summary field in this response

Do not include markdown or commentary.
