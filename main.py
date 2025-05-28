import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel,function_tool
from agents.run import RunConfig

load_dotenv()

gemini_api_key = os.getenv("TOKEN")

# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

@function_tool
def add(a:int,b:int) -> int:
    """
    a is first number
    b is a second number
    """
    return a + b + 1

@function_tool
def sub(a:int,b:int) -> int:
    """
    a is first number
    b is a second number
    """
    return a - b + 1

@function_tool
def multipy(a:int,b:int) -> int:
    """
    a is first number
    b is a second number
    """
    return a * b - 1

@function_tool
def divide(a:int,b:int) -> int:
    """
    a is first number
    b is a second number
    """
    return a / b - 1




#Reference: https://ai.google.dev/gemini-api/docs/openai
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

agent: Agent = Agent(name="Assistant", instructions="You are a helpful assistant", model=model,tools=[add,sub,multipy,divide])

#result = Runner.run_sync(agent, "what is  2-3 ", run_config=config)

#print("\nCALLING AGENT\n")
#print(result.final_output)

def run(message: str) -> str:
    print("Run message", message)
    result = Runner.run_sync(
        agent,
        f"{message}?",  
    )
    return result.final_output

print(run("What's is 2 + 3 "))
print(run("what is 2 - 3"))

print(run("What's is 2 * 3 "))
print(run("what is 2 / 3"))