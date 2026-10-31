
from typing import Any

from utils.model_loader import ModelLoader
from prompt_library.prompt import build_system_prompt
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from utils.memory_retriever import MemoryRetriever
from tools.weather_info_tool import WeatherInfoTool
from tools.place_search_tool import PlaceSearchTool
from tools.expense_calculator_tool import CalculatorTool
from tools.currency_conversion_tool import CurrencyConverterTool


class TripPlannerState(MessagesState):
    verified_memories: str


class GraphBuilder():
    def __init__(
        self, model_provider: str = "groq", feedback_repo: Any | None = None
    ) -> None:
        self.model_loader = ModelLoader(model_provider=model_provider)
        self.llm = self.model_loader.load_llm()
        self.feedback_repo = feedback_repo
        
        self.tools = []
        
        self.weather_tools = WeatherInfoTool()
        self.place_search_tools = PlaceSearchTool()
        self.calculator_tools = CalculatorTool()
        self.currency_converter_tools = CurrencyConverterTool()
        
        self.tools.extend([* self.weather_tools.weather_tool_list, 
                           * self.place_search_tools.place_search_tool_list,
                           * self.calculator_tools.calculator_tool_list,
                           * self.currency_converter_tools.currency_converter_tool_list])
        
        self.llm_with_tools = self.llm.bind_tools(tools=self.tools)
        
        self.graph = None
        
    def agent_function(self, state: TripPlannerState) -> dict[str, list[Any]]:
        """Main agent function"""
        user_question = state["messages"]
        prompt = build_system_prompt(state.get("verified_memories", ""))
        input_question = [prompt] + user_question
        response = self.llm_with_tools.invoke(input_question)
        return {"messages": [response]}

    def memory_retrieval_function(self, state: TripPlannerState) -> dict[str, str]:
        """Attach approved memories that match the current question."""
        if self.feedback_repo is None:
            return {"verified_memories": ""}

        messages = state.get("messages", [])
        latest_message = messages[-1] if messages else ""
        question = getattr(latest_message, "content", latest_message)
        if not isinstance(question, str):
            question = str(question)

        retriever = MemoryRetriever(self.feedback_repo)
        memories = retriever.retrieve_for_question(question, limit=3)
        return {"verified_memories": retriever.format_for_prompt(memories)}

    def build_graph(self):
        graph_builder = StateGraph(TripPlannerState)
        graph_builder.add_node("memory_retrieval", self.memory_retrieval_function)
        graph_builder.add_node("agent", self.agent_function)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_edge(START, "memory_retrieval")
        graph_builder.add_edge("memory_retrieval", "agent")
        graph_builder.add_conditional_edges("agent", tools_condition)
        graph_builder.add_edge("tools", "agent")
        graph_builder.add_edge("agent", END)
        self.graph = graph_builder.compile()
        return self.graph
        
    def __call__(self):
        return self.build_graph()