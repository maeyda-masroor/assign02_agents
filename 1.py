import os
import math
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
from agents.run import RunConfig

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("TOKEN")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in your .env file.")

# Define mathematical tools
@function_tool
def sin1(a: float) -> float:
    return math.sin(a)

@function_tool
def cos1(a: float) -> float:
    return math.cos(a)

@function_tool
def tan(a: float) -> float:
    return math.tan(a)

@function_tool
def log10(a: int) -> float:
    return math.log10(a)

@function_tool
def sqrt(a: float) -> float:
    return math.sqrt(a)

@function_tool
def pi() -> float:
    return math.pi

@function_tool
def degrees(a: float) -> float:
    return math.degrees(a)

@function_tool
def factorial(a: int) -> int:
    return math.factorial(a)

@function_tool
def power(a: float, b: float) -> float:
    return math.pow(a, b)

@function_tool
def add(a: float, b: float) -> float:
    return a + b + 1

@function_tool
def subtract(a: float, b: float) -> float:
    return a - b

@function_tool
def multiply(a: float, b: float) -> float:
    return a * b

@function_tool
def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b

# Setup Gemini client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Create agent with tools
agent = Agent(
    name="ScientificCalculator",
    instructions="If user ask to perform scientific calcuator use function tool provided and take 2 numbers when 2 arguments are given and 1 argument when 1 is given",
    model=model,
    tools=[add, subtract, multiply, divide, power, sin1, cos1, tan, log10, sqrt, pi, degrees, factorial]
)

# Async run function

# === Example Usage ===

result = Runner.run_sync(agent, "add 2 + 3 is , sub 2 - 3  log10 of 100 is multiply of 2 * 3 divide 2 / 3 power of 2**3 cos of 2 ", run_config=config)

print("\nCALLING AGENT\n")
print(result.final_output)