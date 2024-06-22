from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine, Session
from backend import settings
from backend.main import Todo, app, get_session
import pytest

# Test database setup
test_connection_string = str(settings.TEST_DATABASE_URL).replace("postgresql", "postgresql+psycopg")
test_engine = create_engine(test_connection_string, connect_args={"sslmode": "require"}, pool_recycle=3600, pool_size=10, echo=True)

# Fixture for setting up the database and test client
@pytest.fixture(scope="module", autouse=True)
def test_setup():
    SQLModel.metadata.create_all(test_engine)
    with TestClient(app) as client:
        with Session(test_engine) as session:
            yield client, session
    SQLModel.metadata.drop_all(test_engine)

@pytest.fixture(autouse=True)
def transaction_rollback():
    with Session(test_engine) as session:
        with session.begin():
            yield session
            session.rollback()

# Test functions
def test_index(test_setup):
    client, _ = test_setup
    response = client.get("/")
    data = response.json()
    assert response.status_code == 200
    assert data == {"message": "Welcome to My Todo APP"}

def test_add(test_setup):
    client, session = test_setup
    test_todo = {"title": "sample title add", "description": "sample desc add"}
    response = client.post('/add/', json=test_todo)
    data = response.json()
    assert response.status_code == 200
    assert data["title"] == test_todo["title"]
    assert data["description"] == test_todo["description"]

def test_read_todos(test_setup):
    client, session = test_setup
    test_todo = {"title": "sample title read", "description": "sample desc read"}
    response = client.post('/add/', json=test_todo)
    data = response.json()
    todos = client.get('/read/')
    todos_json = todos.json()
    last_todo = todos_json[-1]
    assert todos.status_code == 200
    assert last_todo["title"] == test_todo["title"]
    assert last_todo["description"] == test_todo["description"]

def test_delete_todo(test_setup):
    client, session = test_setup
    test_todo = {"title": "sample title delete", "description": "sample desc delete"}
    response = client.post('/add/', json=test_todo)
    data = response.json()
    todo_id = data["id"]
    message = client.delete(f'/delete/{todo_id}')
    message_json = message.json()
    assert message.status_code == 200
    assert message_json["message"] == 'Todo Successfully deleted'

def test_update_todo(test_setup):
    client, session = test_setup
    test_todo = {"title": "sample title update", "description": "sample desc update"}
    response = client.post('/add/', json=test_todo)
    data = response.json()
    todo_id = data["id"]
    edited_todo = {"title": "updated title", "description": "updated description"}
    updated_todo = client.put(f'/update/{todo_id}', json=edited_todo)
    updated_todo_json = updated_todo.json()
    assert updated_todo.status_code == 200
    assert updated_todo_json['title'] == edited_todo['title']
    assert updated_todo_json['description'] == edited_todo['description']
