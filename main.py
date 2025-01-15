from fastapi import FastAPI
from register_user import router2,internal_user_router
from login_extract_testing import app_router_login

app = FastAPI()

app.include_router(internal_user_router)
app.include_router(router2, prefix="/api")
app.include_router(app_router_login)

@app.get("/")
async def root():
    return {"message": "Welcome to Neovis Chatbot API"}

















