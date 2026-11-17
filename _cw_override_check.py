"""Verify exactly which user messages make critic_gate APPROVE before invoking the critic.

Mirrors agent/continuous_work_critic.py critic_gate() lines ~816-847:
  skip if message starts with '[system:' or '# continuous work'
  APPROVE if lower/strip() == a single-word keyword (exact match)
  APPROVE if any multi-word keyword appears as substring
"""

USER_OVERRIDE_KEYWORDS = {
    "stop",
    "override",
    "enough",
    "done",
    "quit",
    "exit",
    "you can stop",
    "stop working",
    "you are looping",
    "that's enough",
    "we're done",
    "that is enough",
}

CASES = [
    "[CW CRITIC: Your work was REJECTED by the adversarial reviewer.]\n"
    "**Critique:** Critic LLM call failed: Error code: 404 - ... "
    "Required Action: The critic review could not complete. Continue working and try again.",
    "stop",
    "STOP",
    "override",
    "done",
    "you are looping",
    "that's enough",
    "keep going, the critic is broken",
    "# continuous work stop",
]

print(f"{'result':<34} {'skip':<6} {'exact':<6} {'multi':<6} message")
print("-" * 100)
approve_cases = []
for m in CASES:
    cl = m.lower().strip()
    skipped = cl.startswith("[system:") or cl.startswith("# continuous work")
    exact = cl in USER_OVERRIDE_KEYWORDS
    multi = any(kw in cl for kw in USER_OVERRIDE_KEYWORDS if " " in kw)
    if skipped:
        verdict = "SKIPPED -> critic invoked"
    elif exact or multi:
        verdict = "APPROVE (critic never called)"
        approve_cases.append(m)
    else:
        verdict = "critic invoked (fails 404)"
    print(f"{verdict:<34} {str(skipped):<6} {str(exact):<6} {str(multi):<6} {m[:44]!r}")

print()
print("Messages that end the CW loop WITHOUT calling the critic:", [m for m in approve_cases if not m.startswith("#")])
