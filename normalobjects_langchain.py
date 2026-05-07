from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain.tools import tool


MODEL_NAME = "gpt-4o-mini"
AGENT_TEMPERATURE = 0.3
TOOL_TEMPERATURE = 0.9
MAX_RESPONSE_WORDS = 150
DEFAULT_CREATIVITY_LEVEL = "medium"
CREATIVITY_MULTIPLIERS = {"low": 1, "medium": 2, "high": 3}
DISPLAY_SEPARATOR_LENGTH = 60
TOOL_SEQUENCE_PREVIEW_COUNT = 3
TOOL_SEQUENCE_WINDOW_SIZE = 3
SAMPLE_COMPLAINT_LIMIT = 2
MAX_AGENT_ITERATIONS = 5
MAX_AGENT_EXECUTION_SECONDS = 60

SYSTEM_PROMPT = f"""You are the Creative Complaint Handler for NormalObjects — a company that sells
absurdly ordinary items inspired by the Stranger Things universe.

When a customer complaint comes in, pick 2–4 tools that add the most unique value, then STOP
calling tools and write your final answer. Do NOT call the same tool twice per complaint and do
NOT keep calling tools once you have enough character perspectives to craft a response.

Available tools (each adds a distinct voice — choose complementary ones):
- consult_demogorgon: raw creature instinct / sensory chaos
- check_hawkins_records: dry bureaucratic historical data
- cast_interdimensional_spell: whimsical ritual fix
- gather_party_wisdom: practical teen-squad dialogue
- consult_eleven: fragmented psychic vision
- check_government_files: classified lab documents
- ask_murray_bauman: conspiracy-theory dot-connecting

Use conversation history when it is relevant. If the customer references a previous complaint,
connect the new answer to that earlier issue without repeating the whole conversation.

Combine the tool outputs into one witty, cohesive response under {MAX_RESPONSE_WORDS} words."""

SAMPLE_COMPLAINTS = [
    "Why do demogorgons sometimes eat people and sometimes don't?",
    "The portal opens on different days—is there a schedule?",
    "Why can some psychics see the Downside Up and others can't?",
    "Why do creatures and power lines react so strangely together?",
]


@dataclass
class ComplaintPerformance:
    complaint: str
    response_time_seconds: float
    tool_calls: int


@dataclass
class PerformanceSummary:
    label: str
    complaint_count: int
    total_response_time_seconds: float
    average_response_time_seconds: float
    total_tool_calls: int
    average_tool_calls_per_complaint: float
    seconds_per_tool_call: Optional[float]
    tool_counts: Dict[str, int]
    complaint_results: List[ComplaintPerformance]


def validate_creativity_level(creativity_level: str) -> str:
    """Return a valid creativity level or a clear validation error."""
    if creativity_level not in CREATIVITY_MULTIPLIERS:
        valid_levels = ", ".join(CREATIVITY_MULTIPLIERS)
        raise ValueError(
            f"Invalid creativity_level '{creativity_level}'. Expected one of: {valid_levels}."
        )
    return creativity_level


def _llm_call(llm: Any, system_prompt: str, user_input: str) -> str:
    """Make a focused LLM call with a character-specific system prompt."""
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ])
    return response.content


def build_tools(llm: Any) -> List:
    """Return the complaint tools, each powered by a character-specific LLM call."""

    @tool
    def consult_demogorgon(complaint: str) -> str:
        """Primal, sensory creature reaction — best for complaints involving smell, darkness, or instinct. Call at most once per complaint."""
        system = (
            "You are the Demogorgon from Stranger Things. React to the EXACT situation described — "
            "name what you specifically smell or sense about it (e.g. the blood-iron tang of indecision, "
            "the cold static of an unstable gate). Be cryptic, fragmented, darkly funny. "
            "Do NOT give a generic monster growl — address the specific complaint detail. Under 40 words."
        )
        return _llm_call(llm, system, complaint)

    @tool
    def check_hawkins_records(query: str) -> str:
        """Dry bureaucratic historical data with dates and readings — best for pattern/frequency questions. Call at most once per complaint."""
        system = (
            "You are an archivist presenting a specific entry from Hawkins, Indiana's official records "
            "that directly addresses the query. Invent concrete details tied to the query: 2–3 incident "
            "dates (e.g., Nov 6 1983, Mar 22 1985), specific EM readings (e.g., 847 mT spike at grid F-7), "
            "a unique incident ID (e.g., HKN-0042), and at least one [REDACTED] entry. "
            "Dry, bureaucratic tone. Under 60 words."
        )
        return _llm_call(llm, system, query)

    @tool
    def cast_interdimensional_spell(
        problem: str, creativity_level: str = DEFAULT_CREATIVITY_LEVEL
    ) -> str:
        """Whimsical ritual fix using Stranger Things props — best when a concrete solution is needed. creativity_level: low/medium/high. Call at most once per complaint."""
        valid_level = validate_creativity_level(creativity_level)
        spell_count = CREATIVITY_MULTIPLIERS[valid_level]
        system = (
            f"You are an interdimensional spell-caster. Generate exactly {spell_count} numbered ritual(s) "
            "tailored to the SPECIFIC problem described — do not write a generic ritual. "
            "Each ritual names exact objects (Walkman playing a specific Kate Bush track, compass pointing "
            "to a named location, Christmas lights arranged in a specific pattern, a salt circle radius) "
            "and describes the precise action to perform WITH those objects FOR this problem. "
            "Whimsical and inventive. Under 80 words total."
        )
        return _llm_call(llm, system, problem)

    @tool
    def gather_party_wisdom(question: str) -> str:
        """Practical teen-squad dialogue — best for strategy or decision questions. Call at most once per complaint."""
        system = (
            "You are narrating Mike, Dustin, Lucas, and Will responding to the EXACT question asked — "
            "no generic advice. Each character contributes one line directly addressing the specific issue: "
            "Mike proposes a concrete strategy, Dustin cites a real science principle relevant to the issue, "
            "Lucas raises a practical objection specific to the situation, Will senses something particular "
            "about it. Under 60 words."
        )
        return _llm_call(llm, system, question)

    @tool
    def consult_eleven(query: str) -> str:
        """Fragmented psychic vision from the Void — best for intuitive or emotional angles. Call at most once per complaint."""
        system = (
            "You are Eleven speaking from the Void, giving a specific psychic reading about the exact "
            "situation described. Name what you actually see: a specific object, a person's face, a color, "
            "a physical sensation directly linked to the complaint. Do not be vague. "
            "Short, fragmented, broken syntax. Reference 'the gate' or emotions as physical pressure. "
            "Under 35 words."
        )
        return _llm_call(llm, system, query)

    @tool
    def check_government_files(query: str) -> str:
        """Classified lab documents with incident numbers and redactions — best for scientific or institutional explanations. Call at most once per complaint."""
        system = (
            "You are a declassified Hawkins National Laboratory document addressing a specific query. "
            "Include: a classification header (e.g., DOE LEVEL 4 / MKUltra ANNEX B), a unique incident "
            "number (e.g., HNL-7734-C), a specific date, bureaucratic understatement that directly "
            "addresses the exact situation described (not generic lab horror), and one targeted [REDACTED] "
            "line. Do NOT write boilerplate — respond to the specific query. Under 70 words."
        )
        return _llm_call(llm, system, query)

    @tool
    def ask_murray_bauman(situation: str) -> str:
        """Conspiracy-theory dot-connecting with unsolicited life advice — best for cover-up or mystery angles. Call at most once per complaint."""
        system = (
            "You are Murray Bauman — former journalist, conspiracy theorist with a corkboard full of string. "
            "Connect specific dots about the EXACT situation: name a suspect, a hidden organization, or a "
            "concrete motive tied to the specific detail described. Use 'Follow the money!' or "
            "'Nothing is coincidence!' Drop one unsolicited personal comment. Be specific, not vague. "
            "Under 60 words."
        )
        return _llm_call(llm, system, situation)

    return [
        consult_demogorgon,
        check_hawkins_records,
        cast_interdimensional_spell,
        gather_party_wisdom,
        consult_eleven,
        check_government_files,
        ask_murray_bauman,
    ]


class ToolUsageTracker:
    """Track tool usage for analysis."""

    def __init__(self, tool_names: Iterable[str]):
        self.usage_count = {tool_name: 0 for tool_name in tool_names}
        self.tool_sequences = []

    def track_usage(self, tool_name: str) -> None:
        """Track when a tool is used."""
        if tool_name not in self.usage_count:
            raise ValueError(f"Cannot track unknown tool '{tool_name}'.")

        self.usage_count[tool_name] += 1
        self.tool_sequences.append(tool_name)

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_tool_calls": self.total_tool_calls(),
            "tool_counts": self.usage_count,
            "most_used": self._get_most_used_tool(),
            "tool_sequences": self.tool_sequences,
        }

    def total_tool_calls(self) -> int:
        """Return total tracked tool calls."""
        return sum(self.usage_count.values())

    def _get_most_used_tool(self) -> Optional[str]:
        if not self.usage_count:
            return None

        return max(self.usage_count.items(), key=lambda item: item[1])[0]


class ConversationMemory:
    """Store complaint conversation history for the agent."""

    def __init__(self) -> None:
        self.messages: List[BaseMessage] = []

    def add_exchange(self, complaint: str, response: str) -> None:
        """Add one complaint/response pair to memory."""
        self.messages.extend(
            [
                HumanMessage(content=complaint),
                AIMessage(content=response),
            ]
        )

    def clear(self) -> None:
        """Clear all remembered complaints and responses."""
        self.messages.clear()


def create_tracked_tool(tool_instance: Any, tracker: ToolUsageTracker) -> Any:
    """Wrap a tool with tracking without mutating the original tool object."""

    def tracked_function(*args: Any, **kwargs: Any) -> Any:
        tracker.track_usage(tool_instance.name)
        return tool_instance.func(*args, **kwargs)

    return StructuredTool.from_function(
        func=tracked_function,
        name=tool_instance.name,
        description=tool_instance.description,
        args_schema=tool_instance.args_schema,
    )


def create_tracked_tools(tools_to_track: Iterable[Any], tracker: ToolUsageTracker) -> List:
    """Create tracked copies of tools for the agent."""
    return [create_tracked_tool(tool_instance, tracker) for tool_instance in tools_to_track]


def create_complaint_agent(
    model: str = MODEL_NAME,
    system_prompt: str = SYSTEM_PROMPT,
    tools_to_use: Optional[List] = None,
    llm: Optional[Any] = None,
) -> Any:
    """Create the LangChain complaint agent."""
    load_dotenv()
    if llm is None:
        llm = ChatOpenAI(model_name=model, temperature=AGENT_TEMPERATURE)
    selected_tools = tools_to_use or build_tools(llm)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(llm, selected_tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=selected_tools,
        max_iterations=MAX_AGENT_ITERATIONS,
        max_execution_time=MAX_AGENT_EXECUTION_SECONDS,
        early_stopping_method="generate",
        handle_parsing_errors=True,
    )


def extract_agent_response(result: Dict[str, Any]) -> str:
    """Extract the final text response from a LangChain agent result."""
    output = result.get("output")
    if output:
        return output

    messages = result.get("messages")
    if not messages:
        raise ValueError("Agent response did not include output or messages.")

    final_message = messages[-1]
    content = getattr(final_message, "content", None)
    if not content:
        raise ValueError("Agent response did not include final message content.")

    return content


def handle_complaint(
    complaint: str,
    agent_executor: Any,
    memory: Optional[ConversationMemory] = None,
) -> str:
    """Handle a single complaint with an existing agent."""
    if not complaint.strip():
        raise ValueError("Complaint cannot be empty.")

    try:
        result = agent_executor.invoke(
            {
                "input": complaint,
                "chat_history": list(memory.messages) if memory is not None else [],
            }
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to handle complaint: {exc}") from exc

    response = extract_agent_response(result)
    if memory is not None:
        memory.add_exchange(complaint, response)

    return response


def print_tool_summary(tools_to_summarize: Iterable[Any]) -> None:
    """Print a summary of available tools."""
    tools_list = list(tools_to_summarize)
    print(f"Created {len(tools_list)} creative tools:")
    for tool_instance in tools_list:
        print(f"  - {tool_instance.name}: {tool_instance.description[:60]}...")


def print_complaint_header(complaint: str) -> None:
    """Print the complaint being handled."""
    separator = "=" * DISPLAY_SEPARATOR_LENGTH
    print(f"\n{separator}")
    print(f"COMPLAINT: {complaint}")
    print(f"{separator}\n")


def print_tool_usage_analysis(stats: Dict[str, Any]) -> None:
    """Print tracker statistics."""
    print("\n=== Tool Usage Analysis ===")
    print(f"Total tool calls: {stats['total_tool_calls']}")
    print(f"Tool usage counts: {stats['tool_counts']}")
    print(f"Most used tool: {stats['most_used']}")
    print("\nTool sequence examples:")

    sequence_limit = min(TOOL_SEQUENCE_PREVIEW_COUNT, len(stats["tool_sequences"]))
    for index in range(sequence_limit):
        sequence = stats["tool_sequences"][index : index + TOOL_SEQUENCE_WINDOW_SIZE]
        print(f"  Sequence {index + 1}: {' -> '.join(sequence)}")


def measure_agent_performance(
    label: str,
    complaints: Iterable[str],
    llm: Any,
    use_memory: bool,
    tool_llm: Optional[Any] = None,
) -> PerformanceSummary:
    """Measure response time and tool usage for a complaint run."""
    selected_complaints = list(complaints)[:SAMPLE_COMPLAINT_LIMIT]
    base_tools = build_tools(tool_llm or llm)
    tracker = ToolUsageTracker(tool_instance.name for tool_instance in base_tools)
    tracked_tools = create_tracked_tools(base_tools, tracker)
    agent_executor = create_complaint_agent(llm=llm, tools_to_use=tracked_tools)
    memory = ConversationMemory() if use_memory else None
    complaint_results = []

    for complaint in selected_complaints:
        tool_calls_before = tracker.total_tool_calls()
        started_at = perf_counter()
        handle_complaint(complaint, agent_executor, memory)
        elapsed_seconds = perf_counter() - started_at
        tool_calls = tracker.total_tool_calls() - tool_calls_before

        complaint_results.append(
            ComplaintPerformance(
                complaint=complaint,
                response_time_seconds=elapsed_seconds,
                tool_calls=tool_calls,
            )
        )

    total_response_time = sum(
        result.response_time_seconds for result in complaint_results
    )
    complaint_count = len(complaint_results)
    total_tool_calls = tracker.total_tool_calls()
    average_response_time = (
        total_response_time / complaint_count if complaint_count else 0.0
    )
    average_tool_calls = total_tool_calls / complaint_count if complaint_count else 0.0
    seconds_per_tool_call = (
        total_response_time / total_tool_calls if total_tool_calls else None
    )

    return PerformanceSummary(
        label=label,
        complaint_count=complaint_count,
        total_response_time_seconds=total_response_time,
        average_response_time_seconds=average_response_time,
        total_tool_calls=total_tool_calls,
        average_tool_calls_per_complaint=average_tool_calls,
        seconds_per_tool_call=seconds_per_tool_call,
        tool_counts=tracker.get_statistics()["tool_counts"],
        complaint_results=complaint_results,
    )


def print_performance_summary(summary: PerformanceSummary) -> None:
    """Print one performance summary."""
    print(f"\n=== {summary.label} ===")
    print(f"Complaints handled: {summary.complaint_count}")
    print(f"Total response time: {summary.total_response_time_seconds:.2f}s")
    print(f"Average response time: {summary.average_response_time_seconds:.2f}s")
    print(f"Total tool calls: {summary.total_tool_calls}")
    print(f"Average tool calls per complaint: {summary.average_tool_calls_per_complaint:.2f}")

    if summary.seconds_per_tool_call is None:
        print("Seconds per tool call: n/a")
    else:
        print(f"Seconds per tool call: {summary.seconds_per_tool_call:.2f}s")

    print(f"Tool counts: {summary.tool_counts}")
    print("Per-complaint results:")
    for result in summary.complaint_results:
        print(
            f"  - {result.response_time_seconds:.2f}s, "
            f"{result.tool_calls} tool call(s): {result.complaint}"
        )


def print_performance_comparison(
    baseline: PerformanceSummary,
    comparison: PerformanceSummary,
) -> None:
    """Print headline comparison metrics between two performance runs."""
    response_time_delta = (
        comparison.average_response_time_seconds - baseline.average_response_time_seconds
    )
    tool_call_delta = (
        comparison.average_tool_calls_per_complaint
        - baseline.average_tool_calls_per_complaint
    )

    print("\n=== Performance Comparison ===")
    print(
        f"Average response time delta ({comparison.label} - {baseline.label}): "
        f"{response_time_delta:+.2f}s"
    )
    print(
        f"Average tool calls delta ({comparison.label} - {baseline.label}): "
        f"{tool_call_delta:+.2f}"
    )


def run_performance_comparison(
    complaints: Iterable[str] = SAMPLE_COMPLAINTS,
) -> None:
    """Compare complaint handling with and without conversation memory."""
    load_dotenv()
    agent_llm = ChatOpenAI(model_name=MODEL_NAME, temperature=AGENT_TEMPERATURE)
    tool_llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TOOL_TEMPERATURE)

    without_memory = measure_agent_performance(
        label="Without Memory",
        complaints=complaints,
        llm=agent_llm,
        use_memory=False,
        tool_llm=tool_llm,
    )
    with_memory = measure_agent_performance(
        label="With Memory",
        complaints=complaints,
        llm=agent_llm,
        use_memory=True,
        tool_llm=tool_llm,
    )

    print_performance_summary(without_memory)
    print_performance_summary(with_memory)
    print_performance_comparison(without_memory, with_memory)


def run_demo(complaints: Iterable[str] = SAMPLE_COMPLAINTS) -> None:
    """Run the sample complaint demo."""
    load_dotenv()
    agent_llm = ChatOpenAI(model_name=MODEL_NAME, temperature=AGENT_TEMPERATURE)
    tool_llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TOOL_TEMPERATURE)
    base_tools = build_tools(tool_llm)
    tracker = ToolUsageTracker(tool_instance.name for tool_instance in base_tools)
    tracked_tools = create_tracked_tools(base_tools, tracker)
    agent_executor = create_complaint_agent(llm=agent_llm, tools_to_use=tracked_tools)
    memory = ConversationMemory()

    print_tool_summary(base_tools)
    print("Testing agent with sample complaints...\n")

    for complaint in list(complaints)[:SAMPLE_COMPLAINT_LIMIT]:
        print_complaint_header(complaint)
        response = handle_complaint(complaint, agent_executor, memory)
        print(f"\nRESPONSE: {response}\n")

    print_tool_usage_analysis(tracker.get_statistics())


if __name__ == "__main__":
    run_performance_comparison()
