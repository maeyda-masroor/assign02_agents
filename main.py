import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel,function_tool
from agents.run import RunConfig
import math
import nest_asyncio
load_dotenv()


gemini_api_key = os.getenv("TOKEN")

# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")
@function_tool
def sin1(a:float)->float:
    """
    a is sin number
    """
    return math.sin(a)
@function_tool
def cos1(a:float)->float:
    """
    a is cos number
    """
    return math.cos(a)

@function_tool
def tan(a:float)->float:
    """
    a is tan number
    """
    return math.tan(a)

@function_tool
def log10(a:int)->float:
    """
    a is log number
    """
    return math.log10(a)
    

@function_tool
def sqrt(a:int)->int:
    """
    a is sqrt number
    """
    return math.sqrt(a)

@function_tool
def pi()->float:
    return math.pi


@function_tool
def degree(a:int)->float:
    """
    a is degree number
    """
    return math.degree(a)


@function_tool
def factorial(a:int)->float:
    """
    a is factorial number
    """
    return math.factorial(int(a))
    


@function_tool
def pow(a:float , b : float)->float:
    """
        a is base 
        b is power
    """
    return math.pow(a,b)
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
def divide(a:int,b:int) -> float:
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

agent: Agent = Agent(name="Assistant", instructions="You are a helpful assistant", model=model,tools=[add,sub,multipy,divide,pow,sin1,cos1,tan,log10,sqrt,pi,degree,factorial])

#result = Runner.run_sync(agent, "what is  2-3 ", run_config=config)

#print("\nCALLING AGENT\n")
#print(result.final_output)
def run(message: str) -> str:
    print("Run message", message)

    async def _run():
        return await Runner.run(
            agent,
            f"{message}?",
            run_config=config
        )

    result = asyncio.run(_run())
    return result.final_output

print(run("What's is 2 + 3 "))
print(run("what is 2 - 3"))

print(run("What's is 2 * 3 "))
print(run("what is 2.0 / 3.0"))

print(run("what is 2.0**3.0"))
print(run("what is sin(2.0)"))

print(run("What's is cos(2.0) "))
print(run("what is tan(2.0)"))
print(run("what is suqare root of 2 "))

print(run("what is pi"))
print(run("what is degree of 2 "))

print(run("what is factorial of 2 "))
print(run("What's is log10 of 2 "))
