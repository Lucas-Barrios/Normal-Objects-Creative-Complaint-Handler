from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain.tools import tool


MODEL_NAME = "gpt-4o-mini"
MAX_RESPONSE_WORDS = 150
DEFAULT_CREATIVITY_LEVEL = "medium"
CREATIVITY_MULTIPLIERS = {"low": 1, "medium": 2, "high": 3}
DISPLAY_SEPARATOR_LENGTH = 60
TOOL_SEQUENCE_PREVIEW_COUNT = 3
TOOL_SEQUENCE_WINDOW_SIZE = 3
SAMPLE_COMPLAINT_LIMIT = 2

SYSTEM_PROMPT = f"""You are the Creative Complaint Handler for NormalObjects — a company that sells
absurdly ordinary items inspired by the Stranger Things universe.

When a customer complaint comes in, creatively combine your available tools in any order:
- consult_demogorgon: get a chaotic interdimensional perspective
- check_hawkins_records: look up historical Hawkins data
- cast_interdimensional_spell: suggest an imaginative fix
- gather_party_wisdom: tap into the party's collective knowledge
- consult_eleven: get Eleven's psychic reading from the Void
- check_government_files: access classified Hawkins Lab documents
- ask_murray_bauman: hear Murray's conspiracy-theory take on the situation

Use conversation history when it is relevant. If the customer references a previous complaint,
connect the new answer to that earlier issue without repeating the whole conversation.

Use the tools flexibly and combine their outputs into one witty, cohesive response under {MAX_RESPONSE_WORDS} words."""

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
        """Get the Demogorgon's chaotic, instinct-driven perspective on a complaint."""
        system = (
            "You are the Demogorgon from Stranger Things — a creature from the Upside Down. "
            "Respond to the complaint from your primal, hungry perspective. Be cryptic and fragmented, "
            "darkly funny. Reference sensory experiences: smell, darkness, the pull of the gate. "
            "Under 40 words."
        )
        return _llm_call(llm, system, complaint)

    @tool
    def check_hawkins_records(query: str) -> str:
        """Search Hawkins historical records for documented patterns and anomalies."""
        system = (
            "You are an archivist presenting an entry from Hawkins, Indiana's official historical records. "
            "Use a dry, bureaucratic tone. Reference dates, incident counts, and electromagnetic readings. "
            "Include occasional [REDACTED] entries. Under 60 words."
        )
        return _llm_call(llm, system, query)

    @tool
    def cast_interdimensional_spell(
        problem: str, creativity_level: str = DEFAULT_CREATIVITY_LEVEL
    ) -> str:
        """Suggest interdimensional rituals to fix a problem. creativity_level: low/medium/high."""
        valid_level = validate_creativity_level(creativity_level)
        spell_count = CREATIVITY_MULTIPLIERS[valid_level]
        system = (
            f"You are an interdimensional spell-caster. Generate exactly {spell_count} numbered ritual "
            "suggestion(s) for the problem. Each spell must name specific physical objects and actions "
            "with Stranger Things energy: Walkmans, compasses, salt circles, Kate Bush, Christmas lights, "
            "or similar. Be inventive and whimsical. Under 80 words total."
        )
        return _llm_call(llm, system, problem)

    @tool
    def gather_party_wisdom(question: str) -> str:
        """Ask Mike, Dustin, Lucas, and Will for their collective wisdom on a problem."""
        system = (
            "You are narrating Mike, Dustin, Lucas, and Will from Stranger Things giving advice. "
            "Write short character dialogue: Mike is analytical, Dustin is enthusiastic and science-focused, "
            "Lucas is pragmatic, Will is quietly perceptive. Under 60 words."
        )
        return _llm_call(llm, system, question)

    @tool
    def consult_eleven(query: str) -> str:
        """Get Eleven's psychic reading from the Void."""
        system = (
            "You are Eleven from Stranger Things, speaking from the Void. Use short, fragmented speech "
            "with powerful psychic imagery. Reference 'the gate', emotions as physical sensations, and "
            "flashes of vision. Broken syntax. Under 35 words."
        )
        return _llm_call(llm, system, query)

    @tool
    def check_government_files(query: str) -> str:
        """Access classified Hawkins National Laboratory / Department of Energy documents."""
        system = (
            "You are a declassified Hawkins National Laboratory document. Write in classified-file format: "
            "include a classification level (DOE LEVEL 4/5 or MKUltra ANNEX), incident numbers, "
            "bureaucratic understatement about horrifying events, and [REDACTED] entries. Under 70 words."
        )
        return _llm_call(llm, system, query)

    @tool
    def ask_murray_bauman(situation: str) -> str:
        """Hear Murray Bauman's conspiracy-theory take on any situation."""
        system = (
            "You are Murray Bauman — former journalist, conspiracy theorist, vodka enthusiast with a "
            "corkboard full of string and evidence. Respond with barely-contained excitement, connecting "
            "dots others miss, using phrases like 'Follow the money!', 'Nothing is coincidence!', and "
            "dropping unsolicited relationship advice. Under 60 words."
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
        llm = ChatOpenAI(model_name=model)
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
    return AgentExecutor(agent=agent, tools=selected_tools)


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
) -> PerformanceSummary:
    """Measure response time and tool usage for a complaint run."""
    selected_complaints = list(complaints)[:SAMPLE_COMPLAINT_LIMIT]
    base_tools = build_tools(llm)
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
    llm = ChatOpenAI(model_name=MODEL_NAME)

    without_memory = measure_agent_performance(
        label="Without Memory",
        complaints=complaints,
        llm=llm,
        use_memory=False,
    )
    with_memory = measure_agent_performance(
        label="With Memory",
        complaints=complaints,
        llm=llm,
        use_memory=True,
    )

    print_performance_summary(without_memory)
    print_performance_summary(with_memory)
    print_performance_comparison(without_memory, with_memory)


def run_demo(complaints: Iterable[str] = SAMPLE_COMPLAINTS) -> None:
    """Run the sample complaint demo."""
    load_dotenv()
    llm = ChatOpenAI(model_name=MODEL_NAME)
    base_tools = build_tools(llm)
    tracker = ToolUsageTracker(tool_instance.name for tool_instance in base_tools)
    tracked_tools = create_tracked_tools(base_tools, tracker)
    agent_executor = create_complaint_agent(llm=llm, tools_to_use=tracked_tools)
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
