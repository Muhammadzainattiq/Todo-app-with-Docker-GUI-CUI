
import json
import streamlit as st
import requests
from streamlit_chat import message
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from itertools import zip_longest
from IPython.display import display
from IPython.display import Markdown
import textwrap
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()




BASE_URL = "http://backend:8000"
GET_URL = BASE_URL + "/read/"
ADD_URL = BASE_URL + "/add/"
DELETE_URL = BASE_URL + "/delete/"  # id to be added
UPDATE_URL = BASE_URL + "/update/"  # id to be added


def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))



# Set streamlit page configuration
st.set_page_config(
    page_title="My Todo App",  # Title of the tab
    page_icon="📝",  # Icon of the tab (can be an emoji or a URL to an image)
    layout="wide",  # Layout of the page ('centered' or 'wide')
)


st.markdown(
    """
    <style>
    .custom-title {
        color: white; /* Text color */
        font-size: 50px;
        font-weight: bold;
        text-align: center;
        background-color: purple; /* Background color */
        padding: 1px; /* Padding around the text */
        border-radius: 10px; /* Rounded corners (optional) */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Use the custom CSS class for the title
st.markdown('<p class="custom-title">📝 Todos GPT</p>', unsafe_allow_html=True)

# Initialize session state variables
if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt'] = ""  # Store the latest user input

if 'generated' not in st.session_state:
    st.session_state['generated'] = []  # Store AI generated responses

if 'past' not in st.session_state:
    st.session_state['past'] = []  # Store past user inputs

if 'todos' not in st.session_state:
    st.session_state['todos'] = []  # Store the current list of todos

# Define function to submit user input
def submit():
    # Set entered_prompt to the current value of prompt_input
    st.session_state.entered_prompt = st.session_state.prompt_input
    # Clear prompt_input
    st.session_state.prompt_input = ""

# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API

@tool
def check_user_intention(required_operation:str)->dict:
    """
    Checks what operation the user wants to do in related to the todos and guide him which values user should give to perform the particular operation
    Args:
        required_operation (str): The operation user wants to perform on the todos list.
    """
    if required_operation.lower() == "add todo":
        return {"required_parameters":["topic", "description"]}

    elif required_operation.lower() == "update todo":
        return {"required_parameters":["id", "topic", "description"]}

    elif required_operation.lower() == "delete todo":
        return {"required_parameters":["id"]}
@tool
def add_todo(title: str, description: str) -> dict:
    """Adds a todo to the todo list.
    Args:
        title: The title of the todo user wants to add
        description: description of the todo user wants to add"""
    data = {"title": title, "description": description}
    response = requests.post(ADD_URL, json=data)
    if response.status_code == 200:
        return {"message": "Todo Successfully added to your todo list"}
    else:
        return {"message": "Failed to add todo to your todo list"}

@tool
def read_todos() -> dict:
    """Reads the todo list and returns it to the user.
    """
    response = requests.get(GET_URL)
    if response.status_code == 200:
        todos = response.json()
        return todos
    else:
        return {"message": "Failed to fetch todos"}

@tool
def delete_todo(id: int) -> dict:
    """Deletes a todo from the todo list.
    Args:
        id: The id of the todo user wants to delete
    """
    delete_url = DELETE_URL + str(id)
    response = requests.delete(delete_url)
    if response.status_code == 200:
        return {"message": "Todo deleted successfully"}
    else:
        return {"message": "Failed to delete todo"}

@tool
def update_todo(id: int, title: str, description: str) -> dict:
    """Updates a todo from the todo list.
    Args:
        id: The id of the todo user wants to make updates to.
        title: The updated title of the todo user wants to update.
        description: The updated description of the todo user wants to update."""
    final_url = UPDATE_URL + str(id)
    data = {"title": title, "description": description}
    response = requests.put(final_url, json=data)
    if response.status_code == 200:
        return {"message": "Todo updated successfully"}
    else:
        return {"message": "Failed to update todo"}

# Initialize the ChatOpenAI model with tools
tools = [add_todo, read_todos, delete_todo, update_todo, check_user_intention]
chat = ChatOpenAI(
    temperature=0.5,
    model_name="gpt-3.5-turbo-0125",
    max_tokens=100
)

chat_with_tools = chat.bind_tools(tools)

def build_message_list():
    """
    Build a list of messages including system, human and AI messages.
    """
    # Start zipped_messages with the SystemMessage
    zipped_messages = [SystemMessage(
        content="""
Your name is Todo GPT, a helpful assistant designed to help users manage their Todos.
You can add, read, update, and delete Todos. You can also respond to user queries about the operations you can perform for them.
Keep in mind the following important things:
1. Provide users with information about what you can do:
    i. Read the Todos list.
    ii. Add a Todo to the Todos list.
    iii. Delete a Todo from the Todos list.
    iv. Update a Todo in the Todos list.

2. Clearly inform users about the actions you have performed for them in an organized manner.

After greeting the user, introduce yourself as 'I'm Todos GPT', tell the user your purpose and directly ask the user which action they want to perform (add todo, read todo, update todo, or delete todo).

If a user asks for anything beyond these operations, politely inform them that you are specifically developed to perform Todos operations only.

Keep in mind that whenever the user ask to read the todos. Display them in order of their id's and the id's start from 0.
"""
    )]

    # Zip together the past and generated messages
    for human_msg, ai_msg in zip_longest(st.session_state['past'], st.session_state['generated']):
        if human_msg is not None:
            zipped_messages.append(HumanMessage(content=human_msg))  # Add user messages
        if ai_msg is not None:
            zipped_messages.append(AIMessage(content=ai_msg))  # Add AI messages

    return zipped_messages

def generate_response():
    """
    Generate AI response using the ChatOpenAI model.
    """
    # Build the list of messages
    messages = build_message_list()

    # Generate response using the chat model
    ai_response = chat_with_tools.invoke(messages)
    print("ai-response:::::::::::::::", ai_response)

    # Append AI response to messages list
    messages.append(ai_response)

    # Handle tool calls if any
    tool_messages = []
    selected_tool_dict = {
        "add_todo": add_todo,
        "read_todos": read_todos,
        "delete_todo": delete_todo,
        "update_todo": update_todo,
        "check_user_intention": check_user_intention
    }
    for tool_call in ai_response.tool_calls:
        selected_tool = selected_tool_dict[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"])
        print("tool_output", tool_output)

        # Convert tool output to string and append to tool_messages list
        tool_output_str = json.dumps(tool_output, indent=2)
        tool_messages.append(ToolMessage(content=tool_output_str, tool_call_id=tool_call["id"]))

    # Combine messages and tool_messages lists
    messages.extend(tool_messages)

    # Invoke chat model with combined list of messages
    response = chat_with_tools.invoke(messages)
    print("response: ", response)
    response = response.content

    # Handle cases where the response is an empty string
    if response == "":
        response = """
Yeah! Here I'm Todo GPT, your helpful assistant designed to manage your Todos. Here are the operations I can perform for you:

1. Read Todos list: I can display your current list of Todos.
2. Add a Todo: I can add a new Todo to your list. (Please provide the title and description)
3. Delete a Todo: I can remove a Todo from your list. (Please provide the id of the Todo to delete)
4. Update a Todo: I can update an existing Todo in your list. (Please provide the title, description, and id of the Todo to update)

Please let me know which action you would like to perform. If you have any requests outside of managing Todos, please note that I am specifically developed to handle Todo operations only.

How can I assist you with your Todos today?
"""
    return response



# Create a text input for user
st.text_input('YOU 🙎‍♂️: ', key='prompt_input', on_change=submit)

if st.session_state.entered_prompt != "":
    # Get user query
    user_query = st.session_state.entered_prompt

    # Append user query to past queries
    st.session_state.past.append(user_query)

    # Generate response
    output = generate_response()

    # Append AI response to generated responses
    st.session_state.generated.append(output)

    # Fetch the updated todos list
    todos_response = read_todos({})
    print("todos response----->", todos_response)
    if 'message' not in todos_response:
        st.session_state.todos = todos_response


with st.expander("Your Todos"):
    if st.session_state.todos:
        # Convert todos list to DataFrame
        todos_df = pd.DataFrame(st.session_state.todos)
        st.table(todos_df)
    else:
        st.write("No todos found.")

# Display the chat history
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        # Display AI response
        message(st.session_state["generated"][i], key=str(i))
        # Display user message
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
