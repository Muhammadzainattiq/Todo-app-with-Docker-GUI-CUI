from fastapi import FastAPI, Depends, HTTPException
from sqlmodel import create_engine, SQLModel, Field, Session, select, inspect
from typing import Annotated
from backend import settings
from backend.models import Todo, TodoUpdate, TodoCreate
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()


connection_string: str = str(os.getenv("DATABASE_URL")).replace("postgresql", "postgresql+psycopg")
engine = create_engine(connection_string, connect_args={"sslmode": "disable"}, pool_recycle=3600, pool_size=10, echo=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Creating Tables")
    create_tables()
    print("Tables Created")
    try:
        yield
    finally:
        print("Lifespan context ended")


app = FastAPI(lifespan=lifespan)


def create_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


@app.get('/')
def index():
    return {"message": "Welcome to My Todo APP"}

@app.post('/add/')
def add_todo(todo: Todo, session: Annotated[Session, Depends(get_session)]):
    session.add(todo)
    session.commit()
    session.refresh(todo)
    return todo


@app.get('/read/', response_model=list[Todo])
def get_todo(session: Annotated[Session, Depends(get_session)]):
    statement = select(Todo)
    todos = session.exec(statement).all()
    return todos

@app.delete("/delete/{id}")
def delete_todo(id: int, session: Annotated[Session, Depends(get_session)]):
    delete_todo = session.exec(select(Todo).where(Todo.id == id)).first()
    if delete_todo:
        session.delete(delete_todo)
        session.commit()
        return {'message': 'Todo Successfully deleted'}
    else:
        raise HTTPException(status_code=404, detail="Todo not found")

@app.put('/update/{id}')
def update_todo(id: int, updated_todo: TodoUpdate, session: Annotated[Session, Depends(get_session)]):
    statement = select(Todo).where(Todo.id == id)
    db_todo = session.exec(statement).first()
    if db_todo:
        if updated_todo.title is not None:
            db_todo.title = updated_todo.title
        if updated_todo.description is not None:
            db_todo.description = updated_todo.description
        session.commit()
        session.refresh(db_todo)
        return db_todo
    else:
        raise HTTPException(status_code=404, detail="Todo not found")
