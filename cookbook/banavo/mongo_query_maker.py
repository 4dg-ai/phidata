"""Run `pip install lancedb` to install dependencies."""

from phi.embedder.ollama import OllamaEmbedder
from phi.agent import Agent
from phi.model.ollama import Ollama
from phi.tools.file import FileTools
import json
from schema import collections
from phi.assistant import Message
from phi.llm.openai import OpenAIChat
from phi.llm.groq import Groq
from textwrap import dedent
from phi.assistant import Assistant


# Configure the language model
#model = Ollama(model="llama3.2:latest", temperature=0.0)

# Create Ollama embedder
embedder = OllamaEmbedder(model="nomic-embed-text", dimensions=768)

additional_message = {
        "role": "system",
        "content": f"Use this as the database schema: {collections}"
    }

llm_os = Assistant(
        name="mongo_db_query_maker",
        user_id="user",
        llm=OpenAIChat(model="gpt-4o-mini"),
        description=dedent(
            """\
            You are the best at converting natural language queries to no sequel queries.
            You have an excellent grasp of natural language and no-sql, so you can modify the 
            natural language query as needed to match the schema in the database.\
            """
        ),
        instructions=[
            "The user may give you only the natural language query. Make use of the database schema to provide the no-sql query corresponding to the natural language query.", 
            "Provide only the query, you need not explain anything. Keep in mind that this is a NO SEQUEL DATABASE.",
        ],
        show_tool_calls=True,
        read_chat_history=True,
        add_chat_history_to_messages=True,
        num_history_messages=20,
        markdown=True,
        additional_messages=[additional_message],
    )

# Use the agent to generate and print a response to a query, formatted in Markdown
# agent.print_response(f"Generate a mongodb query for the input: 'What is the total net sales in Spetember 2024'.", markdown=True)
response = ""
for delta in llm_os.run("Which are the top selling products in September 2024?"):
    response += delta
print(response)