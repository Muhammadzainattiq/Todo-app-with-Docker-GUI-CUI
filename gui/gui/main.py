import streamlit as st
import requests
import pandas as pd

BASE_URL = "http://backend:8000"
GET_URL = BASE_URL + "/read/"
ADD_URL = BASE_URL + "/add/"
DELETE_URL = BASE_URL + "/delete/" # id to be added
UPDATE_URL = BASE_URL + "/update/" # id to be added

st.set_page_config(
    page_title="My Todo App",  # Title of the tab
    page_icon="📝",  # Icon of the tab (can be an emoji or a URL to an image)
    layout="wide",  # Layout of the page ('centered' or 'wide')
)

def fetch_todos():
    response = requests.get(GET_URL)
    if response.status_code == 200:
        todos = response.json()
        return todos
    else:
        st.error("Failed to fetch todos")
        return []

def display_todos(placeholder):
    todos = fetch_todos()
    todos_data = [{"Id": todo['id'], "Title": todo['title'], "Description": todo['description']} for todo in todos]
    df = pd.DataFrame(todos_data)
    # df = df.sort_values(by="id") 
    df = df.reset_index(drop=True)
    with placeholder.container():
        st.table(df)

def add_todo(title, desc, placeholder):
    data = {"title": title, "description": desc}
    with st.spinner("Adding..."):
        response = requests.post(ADD_URL, json=data)
        if response.status_code == 200:
            st.success("Todo added successfully")
        else:
            st.error("Failed to add todo")
    display_todos(placeholder)

def delete_todo(id, placeholder):
    final_url = DELETE_URL + str(id)
    with st.spinner("Deleting..."):
        response = requests.delete(final_url)
        if response.status_code == 200:
            st.success("Todo deleted successfully")
        else:
            st.error("Failed to delete todo")
    display_todos(placeholder)

def update_todo(id, title, desc, placeholder):
    final_url = UPDATE_URL + str(id)
    data = {"title": title, "description": desc}
    with st.spinner("Updating..."):
        response = requests.put(final_url, json=data)
        if response.status_code == 200:
            st.success("Todo updated successfully")
        else:
            st.error("Failed to update todo")
    display_todos(placeholder)

def main():
    st.markdown("""
        <style>
        .main-title {
            font-size: 3em;
            color: #4A90E2;
            text-align: center;
            font-family: 'Arial', sans-serif;
        }
        </style>
        <h1 class="main-title">📝My Todo App</h1>
    """, unsafe_allow_html=True)
    todos_placeholder = st.empty()
   
    
    # Display the current todos in the placeholder
    display_todos(todos_placeholder)

    st.markdown("""
    <style>
    .sub-title {
        font-size: 2em;
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    </style>
    <h2 class="sub-title">➕ Add Todo:</h2>
""", unsafe_allow_html=True)
    

    title = st.text_input("Enter Title: ")
    desc = st.text_area("Enter Description: ")
    if st.button("Add"):
        add_todo(title, desc, todos_placeholder)

    st.markdown("""
    <style>
    .sub-title {
        font-size: 2em;
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    </style>
    <h2 class="sub-title">➖ Delete Todo:</h2>
""", unsafe_allow_html=True)



    delete_id = st.text_input("Enter Id to Delete: ")
    if st.button("Delete"):
        delete_todo(delete_id, todos_placeholder)

    st.markdown("""
    <style>
    .sub-title {
        font-size: 2em;
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    </style>
    <h2 class="sub-title">✏️ Update Todo:</h2>
""", unsafe_allow_html=True)


    update_id = st.text_input("Enter Id: ")
    update_title = st.text_input("Enter Title To Update: ")
    update_desc = st.text_area("Enter Description To Update")
    if st.button("Update"):
        update_todo(update_id, update_title, update_desc, todos_placeholder)

if __name__ == "__main__":
    main()
