import random
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from typing import List

load_dotenv()

# --- Tools ---

@tool
def consult_demogorgon(complaint: str) -> str:
    """Get the Demogorgon's perspective on a complaint about the Upside Down.

    The Demogorgon is a creature from the Upside Down. It might have insights
    about interdimensional inconsistencies, but its perspective is... unique.

    Args:
        complaint: The complaint about the Upside Down

    Returns:
        The Demogorgon's perspective (creative and possibly chaotic)
    """
    responses = [
        f"The Demogorgon tilts its head. It seems confused by '{complaint}'. Perhaps the issue is that you're thinking in three dimensions?",
        f"The Demogorgon makes a sound that might be agreement. It suggests that the problem might be temporal - things work differently in the Upside Down's time.",
        f"The Demogorgon appears to be eating something. It doesn't seem to understand the concept of '{complaint}' - maybe consistency isn't a priority there?"
    ]
    return random.choice(responses)

@tool
def check_hawkins_records(query: str) -> str:
    """Search Hawkins historical records for information.

    Hawkins, Indiana has a long history of strange occurrences. These records
    might contain clues about patterns or explanations.

    Args:
        query: What to search for in the records

    Returns:
        Information from Hawkins historical records
    """
    records = {
        "portal": "Records show portals have opened on various dates with no clear pattern. Weather, electromagnetic activity, and unknown factors seem involved.",
        "monsters": "Historical records indicate creatures from the Upside Down behave differently based on environmental factors, time of day, and proximity to certain individuals.",
        "psychics": "Records show that psychic abilities vary greatly. Some individuals can move objects but not see the future, others can see visions but not move things.",
        "electricity": "Hawkins has a history of electrical anomalies. Records suggest a connection between the Upside Down and electromagnetic fields."
    }

    for key, value in records.items():
        if key in query.lower():
            return value

    return f"Records don't contain specific information about '{query}', but they note that many unexplained events have occurred in Hawkins over the years."

@tool
def cast_interdimensional_spell(problem: str, creativity_level: str = "medium") -> str:
    """Suggest a creative interdimensional spell to fix a problem.

    Sometimes the best solution is a creative one that doesn't follow normal rules.
    This tool suggests imaginative fixes for Upside Down problems.

    Args:
        problem: The problem to solve
        creativity_level: How creative to be (low, medium, high)

    Returns:
        A creative spell or solution suggestion
    """
    creativity_multiplier = {"low": 1, "medium": 2, "high": 3}[creativity_level]

    spells = [
        f"Try chanting 'Bemca Becma Becma' three times while holding a Walkman. This might recalibrate the interdimensional frequencies related to: {problem}",
        f"Create a salt circle and place a compass in the center. The magnetic anomalies might help stabilize: {problem}",
        f"Play 'Running Up That Hill' backwards at the exact location of the issue. The temporal resonance could fix: {problem}",
        f"Gather three items: a lighter, a compass, and something personal. Arrange them in a triangle while thinking about: {problem}. The emotional connection might help.",
    ]

    selected = random.sample(spells, min(creativity_multiplier, len(spells)))
    return "\n".join(selected)

@tool
def gather_party_wisdom(question: str) -> str:
    """Ask the D&D party (Mike, Dustin, Lucas, Will) for their collective wisdom.

    The party has solved many mysteries together. Their combined knowledge
    and different perspectives can provide insights.

    Args:
        question: The question or problem to ask the party about

    Returns:
        The party's collective wisdom and suggestions
    """
    party_responses = {
        "portal": "Mike: 'Portals are unpredictable, but they usually open near strong emotional events or electromagnetic disturbances.' Dustin: 'Also, they seem to follow some kind of pattern related to the Mind Flayer's activity.'",
        "monsters": "Lucas: 'Demogorgons are territorial but also opportunistic.' Will: 'They can sense fear and strong emotions. Maybe that's why they act differently sometimes.'",
        "psychics": "Mike: 'El's powers seem connected to her emotional state.' Dustin: 'And they're limited by her physical and mental energy. That's probably why she can't do everything.'",
        "electricity": "Lucas: 'The Upside Down seems to interfere with electrical systems.' Dustin: 'But it also creates strange connections. It's like a feedback loop.'"
    }

    for key, response in party_responses.items():
        if key in question.lower():
            return response

    return "The party huddles together. Mike: 'This is a tough one.' Dustin: 'We need more information.' Lucas: 'Let's think about what we know.' Will: 'Maybe we should consult other sources?'"

# --- Agent ---

SYSTEM_PROMPT = """You are the Creative Complaint Handler for NormalObjects — a company that sells
absurdly ordinary items inspired by the Stranger Things universe.

When a customer complaint comes in, creatively combine your available tools in any order:
- consult_demogorgon: get a chaotic interdimensional perspective
- check_hawkins_records: look up historical Hawkins data
- cast_interdimensional_spell: suggest an imaginative fix
- gather_party_wisdom: tap into the party's collective knowledge

Use the tools flexibly and combine their outputs into one witty, cohesive response under 150 words."""

tools: List = [
    consult_demogorgon,
    check_hawkins_records,
    cast_interdimensional_spell,
    gather_party_wisdom
]

agent_executor = create_agent(
    model="openai:gpt-4o-mini",
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)

print(f"Created {len(tools)} creative tools:")
for t in tools:
    print(f"  - {t.name}: {t.description[:60]}...")

# --- Tool Usage Tracker ---

class ToolUsageTracker:
    """Track tool usage for analysis"""
    def __init__(self):
        self.usage_count = {tool.name: 0 for tool in tools}
        self.tool_sequences = []

    def track_usage(self, tool_name: str):
        """Track when a tool is used"""
        if tool_name in self.usage_count:
            self.usage_count[tool_name] += 1
            self.tool_sequences.append(tool_name)

    def get_statistics(self):
        """Get usage statistics"""
        return {
            "total_tool_calls": sum(self.usage_count.values()),
            "tool_counts": self.usage_count,
            "most_used": max(self.usage_count.items(), key=lambda x: x[1])[0] if self.usage_count else None,
            "tool_sequences": self.tool_sequences
        }

tracker = ToolUsageTracker()

# Patch each tool's invoke method so the tracker captures every agent call
for t in tools:
    _orig_invoke = t.invoke
    _name = t.name
    def _make_wrapper(name, orig):
        def wrapper(input, **kwargs):
            tracker.track_usage(name)
            return orig(input, **kwargs)
        return wrapper
    t.invoke = _make_wrapper(_name, _orig_invoke)

# --- Complaint Handler ---

def handle_complaint(complaint: str) -> str:
    """Handle a single complaint"""
    print(f"\n{'='*60}")
    print(f"COMPLAINT: {complaint}")
    print(f"{'='*60}\n")
    result = agent_executor.invoke({"messages": [{"role": "user", "content": complaint}]})
    return result["messages"][-1].content

# --- Main ---

if __name__ == "__main__":
    complaints = [
        "Why do demogorgons sometimes eat people and sometimes don't?",
        "The portal opens on different days—is there a schedule?",
        "Why can some psychics see the Downside Up and others can't?",
        "Why do creatures and power lines react so strangely together?",
    ]

    print("Testing agent with sample complaints...\n")
    for complaint in complaints[:2]:
        response = handle_complaint(complaint)
        print(f"\nRESPONSE: {response}\n")

    print("\n=== Tool Usage Analysis ===")
    stats = tracker.get_statistics()
    print(f"Total tool calls: {stats['total_tool_calls']}")
    print(f"Tool usage counts: {stats['tool_counts']}")
    print(f"Most used tool: {stats['most_used']}")
    print(f"\nTool sequence examples:")
    for i in range(min(3, len(stats['tool_sequences']))):
        print(f"  Sequence {i+1}: {' -> '.join(stats['tool_sequences'][i:i+3])}")
