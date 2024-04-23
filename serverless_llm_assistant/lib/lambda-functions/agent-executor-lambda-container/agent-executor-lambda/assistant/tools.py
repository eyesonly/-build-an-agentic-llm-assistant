import boto3
from langchain.agents import Tool
from langchain.llms.bedrock import Bedrock
from langchain_community.tools import DuckDuckGoSearchRun
from .calculator import CustomCalculatorTool
from .config import AgenticAssistantConfig
from .rag import get_rag_chain

config = AgenticAssistantConfig()
bedrock_runtime = boto3.client("bedrock-runtime", region_name=config.bedrock_region)

claude_llm = Bedrock(
    model_id=config.llm_model_id,
    client=bedrock_runtime,
    model_kwargs={"max_tokens_to_sample": 500, "temperature": 0.0},
)

search = DuckDuckGoSearchRun()
custom_calculator = CustomCalculatorTool()
rag_qa_chain = get_rag_chain(config, claude_llm, bedrock_runtime)

LLM_AGENT_TOOLS = [
    Tool(
        name="Search",
        func=search.invoke,
        description=(
            "Use when you need to answer questions about current events, news or people."
            " You should ask targeted questions."
        ),
    ),
    Tool(
        name="Calculator",
        func=custom_calculator,
        description=(
            "Always Use this tool when you need to answer math questions."
            " The input to Calculator can only be a valid math expression, such as 55/3."
        ),
    ),
    Tool(
        name="SemanticSearch",
        func=lambda query: rag_qa_chain({"question": query}),
        description=(
            "Use when you are asked questions about financial reports of companies."
            " The Input should be a correctly formatted question."
        ),
    )
]

# # This module will be edited in Lab 03 to add the agent tools.
# import boto3
# from langchain.agents import Tool
# from langchain.llms.bedrock import Bedrock
# from langchain_community.tools import DuckDuckGoSearchRun
# from .calculator import CustomCalculatorTool
# from .config import AgenticAssistantConfig

# config = AgenticAssistantConfig()
# bedrock_runtime = boto3.client("bedrock-runtime", region_name=config.bedrock_region)

# claude_llm = Bedrock(
#     model_id=config.llm_model_id,
#     client=bedrock_runtime,
#     model_kwargs={"max_tokens_to_sample": 500, "temperature": 0.0},
# )

# search = DuckDuckGoSearchRun()
# custom_calculator = CustomCalculatorTool()

# LLM_AGENT_TOOLS = [
#     Tool(
#         name="Search",
#         func=search.invoke,
#         description=(
#             "Use when you need to answer questions about current events, news or people."
#             " You should ask targeted questions."
#         ),
#     ),
#     Tool(
#         name="Calculator",
#         func=custom_calculator,
#         description=(
#             "Always Use this tool when you need to answer math questions."
#             " The input to Calculator can only be a valid math expression, such as 55/3."
#         ),
#     ),
# ]