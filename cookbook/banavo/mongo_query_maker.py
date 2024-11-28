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
        llm=OpenAIChat(model="gpt-4o-mini", temperature=0.3),
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
            "Ensure that you are using the correct collections and fields for every query.",
            "Some natural language queries may require you to write pipelines to fetch data from multiple collections.",
            "Ensure that you make use of all available data in order to provide the perfect query to the mongodb instance everytime.",
            "There's a reward for you everytime you give the correct query.",
            #"For any natural language query, if it is not possible to construct a no sequel query with the schema provided, PLEASE MENTION SO",
        ],
        show_tool_calls=True,
        read_chat_history=True,
        add_chat_history_to_messages=True,
        num_history_messages=20,
        markdown=True,
        add_datetime_to_instructions = True,
        additional_messages=[additional_message],
    )

questions = [
    "What was the best selling item last year on Black Friday?",
    "What stone will be best to sell this year on Black Friday?",
    "What are the top 5 best selling products in the Silver Necklace product category?",
    "What is the best-selling item today?",
    "How much of sales came from 1 year dormant customers?",
    "Yesterday, how much of sales came from 1 year dormant customers?",
    "Did those customers use any code?",
    "Yesterday, did the dormant customers use any promo code?",
    "What price point is best suitable for Black Friday sales?",
    "What price range is most suitable for Black Friday sales?",
    "What price range of items has better conversion rate in the Black Friday season period?",
    "Which price range has more conversion on website and mobile in the Black Friday season?",
    "What is the number of orders cancelled within 10 secs in the last month and what is the total amount of those cancelled orders?",
    "From which sales source maximum cancellation is happening?",
    "What is the growth on digital platforms year on year as compared to last year?",
    "From which promo code maximum sales conversion is coming?",
    "What is the conversion rate of customers which are coming from sweepstakes?",
    "What is the conversion rate of customers which are coming from sweepstakes in the last 6 months?",
    "How much sales contribution in percentage is coming from lab grown diamonds in the last 1 month?",
    "Which shape in the lab grown is the most selling?",
    "What percentage of auction participants are new versus returning customers?",
    "What is the Average Selling Price (ASP) for items sold through auctions?",
    "Is it a good strategy to post images on TikTok?",
    "Help me create an organic content strategy for TikTok. Please include the type of format for the content as well.",
    "What are effective TikTok content formats for organic growth?",
    "How to develop a TikTok content strategy for organic reach?",
    "When creating a campaign on Meta what should be the ideal audience to target?",
    "Give me 5 products that I can promote on Facebook.",
    "Suggest 5 products suitable for Meta campaigns.",
    "What are the trending keywords on Google for Black Friday?",
    "Compare the sales performance of Meta for 2023 and 2024.",
    "What is the distribution of sales across different sales channels?",
    "What is the distribution of sales across Google and Meta?",
    "What is the month wise sales and return percentage on loose lab grown diamond in the last 3 months?",
    "What is best-seller item today?",
    "Tell me which products today outperformed the set target.",
    "Searching for products by category name that outperformed their targets today?",
    "Searching for products by category name that exceeded their targets this week?",
    "Give top selling products in last 7 days.",
    "Give me top selling loose gemstones from last 7 days on web.",
    "Give me top selling products from loose gemstone category on web in last 7 days.",
    "Which are top 3 categories on FPC, in terms of revenue, in last 7 days?",
    "Other than lab grown diamond, give me top 5 selling products in loose gemstones category.",
    "Do you have access to last year data?",
    "Share how many products got returned by customers in the month of August.",
    "How much value of products were returned in the month of August, give me dollar figure.",
    "What are top 5 performing Tanzanite SKUs in loose gems category on web?",
    "Are there specific price ranges that lead to higher sales volumes?",
    "Are there specific price ranges that lead to higher sales volumes for loose gems category?",
    "What impact do discounts or promotional offers have on the sales of loose gemstones?",
    "What are the top three most popular gemstones sold on this website?",
    "What are the primary demographics of customers purchasing loose gemstones?",
    "Do loose gemstones sell better via the website or other channels like live TV auctions?",
    "What is growth for Thanksgiving days, Black Friday & Cyber Monday in last 5 years?",
    "What is growth for Thanksgiving days, Black Friday & Cyber Monday in last 5 years for ShopLC?",
    "What is growth percentage for Thanksgiving day, Black Friday & Cyber Monday for last 2 years?",
    "Give me sales growth Black Friday 2022 versus 2023.",
    "Please share last year Black Friday sales.",
    "Please share sales for Black Friday 2023.",
    "What is web sales and contribution percentage in total ShopLC sales for 2023-24?",
    "What is web and mobile sales and contribution percentage in total ShopLC sales for 2023-24?",
    "What is PNP per PC for RA for last year?",
    "Please share all channel wise sales for last year.",
    "What is online auction sales for last year?",
    "What is RA sales for last year?",
    "What is mobile sales for last year?",
    "What is mobile sales for 2023-24?",
    "What is mobile sales for April 2023 to March 2024?",
    "What is web and mobile sales for Black Friday 2023?",
    "What is web and mobile total sales for Black Friday 2023?",
    "What is total sales for April 2023 to March 2024?",
    "What is $1 auction sales for April 2023 to March 2024?",
    "What is total sales for Black Friday 2023?",
    "What is sales for Black Friday 2022 and 2023?",
    "What is average sales for Sunday for April 2024 to October 2024?",
    "What is average sales for 22 hour in October 2024?",
    "What is average sales for 10pm from April 2024 to October 2024?",
    "Please share all under $10 event dates for 2024.",
    "Which color of Murano is selling well in 2024?",
    "Last 7 Friday sales with date.",
    "Top 10 best selling stones in current year.",
    "Highest sales month in current year."
]

# Use the agent to generate and print a response to a query, formatted in Markdown
# agent.print_response(f"Generate a mongodb query for the input: 'What is the total net sales in Spetember 2024'.", markdown=True)
responses= {}
for idx, question in enumerate(questions):
    response = ""

    # Simulating the LLM response collection
    # Replace `llm_os.run()` with your actual LLM API call
    for delta in llm_os.run(question):  
        response += delta
    
    # Store the response in the dictionary
    responses[question] = response
    print(f"Processed question {idx + 1}/{len(questions)}")

# Save the responses to a JSON file
with open("nosql_queries.json", "w") as file:
    json.dump(responses, file, indent=4)