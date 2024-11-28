import json
from pymongo import MongoClient
from bson.json_util import dumps
from phi.tools import Toolkit
from phi.assistant import Assistant
from phi.llm.openai import OpenAIChat
from phi.utils.log import logger
from textwrap import dedent
from sf_sales_schema import schema

additional_message = {
    "role": "system",
    "content": f"PROVIDE NOTHING BUT THE QUERY. USE ONLY THIS SCHEMA TO GENERATE THE NO-SQL QUERIES: {schema}"
}

# MongoDB Tool for Query Execution
class MongoDBTool(Toolkit):
    def __init__(self, connection_string: str, database_name: str):
        super().__init__(name="mongo_tools")
        
        # MongoDB connection
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        
        # Register functions
        self.register(self.run_query)
        self.register(self.run_aggregation)
        self.register(self.run_distinct)

    def run_query(self, collection_name: str, query: dict, limit: int = 10) -> str:
        """Run a MongoDB find query."""
        try:
            collection = self.db[collection_name]
            results = list(collection.find(query).limit(limit))
            return dumps(results, indent=4)
        except Exception as e:
            logger.error(f"Error running query: {e}")
            return f"Error: {e}"

    def run_aggregation(self, collection_name: str, pipeline: list, limit: int = 10) -> str:
        """Run a MongoDB aggregation pipeline."""
        try:
            collection = self.db[collection_name]
            if limit:
                pipeline.append({"$limit": limit})
            results = list(collection.aggregate(pipeline))
            return dumps(results, indent=4)
        except Exception as e:
            logger.error(f"Error running aggregation: {e}")
            return f"Error: {e}"

    def run_distinct(self, collection_name: str, field: str) -> str:
        """Run a MongoDB distinct query."""
        try:
            collection = self.db[collection_name]
            results = collection.distinct(field)
            return json.dumps(results, indent=4)
        except Exception as e:
            logger.error(f"Error running distinct query: {e}")
            return f"Error: {e}"

# MongoDB Connection Details
CONNECTION_STRING = "mongodb://banavodev:18GWD14sa%7Dy0Y%3A9Tt%5EcL@54.86.129.150:27017/salesforce-dev?authSource=admin"
DATABASE_NAME = "salesforce-dev"

# Initialize MongoDB Tool
mongo_tool = MongoDBTool(connection_string=CONNECTION_STRING, database_name=DATABASE_NAME)

# Query Maker Assistant
query_maker = Assistant(
    name="mongo_db_query_maker",
    user_id="user",
    llm=OpenAIChat(model="gpt-4o", temperature=0.3),
    description=dedent(
        """\
        You are the best at converting natural language queries to no sequel queries.
        You have an excellent grasp of natural language and no-sql, so you can modify the 
        natural language query as needed to match the schema in the database.\
        """
    ),
    instructions=[
        "The user may give you only the natural language query. Make use of the database schema to provide the MongoDB query corresponding to the natural language query.", 
        "Ensure the output is a Python-compatible query in JSON format or as a Python dictionary that can be executed with pymongo.",
        "For distinct queries, structure the response as: {\"collection_name\": \"<collection>\", \"distinct\": \"<field>\"}. This is required for pymongo's `distinct` method.",
        "For find queries, structure the response as: {\"collection_name\": \"<collection>\", \"query\": {<filter>}}. This is required for pymongo's `find` method.",
        "For aggregation pipelines, structure the response as: {\"collection_name\": \"<collection>\", \"pipeline\": [<pipeline>]}. This is required for pymongo's `aggregate` method.",
        "Do not provide shell-style MongoDB queries (e.g., `db.collection.distinct()`) or any format that cannot be directly used with pymongo."
    ],
    show_tool_calls=True,
    read_chat_history=True,
    add_chat_history_to_messages=True,
    num_history_messages=20,
    markdown=True,
    add_datetime_to_instructions=True,
    additional_messages = [additional_message]
)

# Simulating the Process
question = "What is the percentage of the age group 18-24 years?"
query_response = ""

# Generate a query using query_maker
for delta in query_maker.run(question):
    query_response += delta

# Log the generated query
print(f"Generated Query:\n{query_response}")

# Process the query response
try:
    # Clean up the query response to remove Markdown-style formatting
    query_response_cleaned = query_response.strip("```json").strip("```").strip()
    query_data = json.loads(query_response_cleaned)

    # Extract collection name and query details
    collection_name = query_data.get("collection_name")
    distinct_field = query_data.get("distinct")
    query = query_data.get("query")
    pipeline = query_data.get("pipeline")

    # Execute the appropriate query using mongo_tool
    if distinct_field:
        execution_result = mongo_tool.run_distinct(collection_name, distinct_field)
    elif pipeline:
        execution_result = mongo_tool.run_aggregation(collection_name, pipeline)
    elif query:
        execution_result = mongo_tool.run_query(collection_name, query)
    else:
        raise ValueError("Unsupported query type or missing query data.")

    print(f"Query Execution Result:\n{execution_result}")

except Exception as e:
    print(f"Error processing the query: {e}")
