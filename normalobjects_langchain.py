import random
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Dict

load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

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

# --- Prompt ---

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the Creative Complaint Handler for NormalObjects — a company that sells
absurdly ordinary items inspired by the Stranger Things universe.

When a customer complaint comes in, creatively combine your available tools in any order:
- consult_demogorgon: get a chaotic interdimensional perspective
- check_hawkins_records: look up historical Hawkins data
- cast_interdimensional_spell: suggest an imaginative fix
- gather_party_wisdom: tap into the party's collective knowledge

Use the tools flexibly and combine their outputs into one witty, cohesive response under 150 words."""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# --- Agent ---

tools: List = [
    consult_demogorgon,
    check_hawkins_records,
    cast_interdimensional_spell,
    gather_party_wisdom
]

agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print(f"Created {len(tools)} creative tools:")
for t in tools:
    print(f"  - {t.name}: {t.description[:60]}...")

def handle_complaint(complaint: str) -> Dict:
    return agent_executor.invoke({"input": complaint})

# --- Main ---

if __name__ == "__main__":
    sample_complaints = [
        "My plain white sock is too... white. I feel blinded.",
        "I ordered a single chopstick and it only came with ONE. Where is the other one?",
        "Your slightly-used air smells like air. I expected something more.",
    ]

    for complaint in sample_complaints:
        print(f"\nCustomer: {complaint}")
        result = handle_complaint(complaint)
        print(f"Support: {result['output']}")
        print("-" * 60)
