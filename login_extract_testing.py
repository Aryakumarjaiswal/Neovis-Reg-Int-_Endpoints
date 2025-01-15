

from fastapi import FastAPI, HTTPException, APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from register_user import Session_Table
from Database import SessionLocal
import httpx
import uuid
from datetime import datetime

# Define the API router
app_router_login = APIRouter()

# Define your constants
BASE_URL = "https://shark-app-6wiyn.ondigitalocean.app/api/v1"
LOGIN_ENDPOINT = f"{BASE_URL}/auth/login"


# Pydantic model for the login request
class LoginRequest(BaseModel):
    email: str
    password: str
# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app_router_login.post("/fetch_user_details/")
async def fetch_user_details(credentials: LoginRequest, db: Session = Depends(get_db)):
    """
    Logs in with the provided email and password and retrieves the user details (ID, ROLE).
    Adds an entry in the Session_Table if data is extracted successfully.
    """
    async with httpx.AsyncClient() as client:
       
        print(credentials.dict())
        login_response = await client.post(LOGIN_ENDPOINT, json=credentials.dict())

       
        if login_response.status_code != 201:
            raise HTTPException(
                status_code=login_response.status_code,
                detail=f"Login failed: {login_response.text}",
            )

        # Parse the login response
        data = login_response.json()
        token = data.get("token")
        user_id = data["user"]["id"]
        user_role = data["user"]["user_role"]["role"]

        # Ensure all required data is present
        if not token:
            raise HTTPException(status_code=400, detail="Token not found in the response.")
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID not found in the response.")
        if not user_role:
            raise HTTPException(status_code=400, detail="User role not found in the response.")

     
        try:
        
          

            new_session = Session_Table(
               
                user_id=user_id,
               user_type=user_role,
                status="active",  # Default status
                started_at= datetime.utcnow().strftime('%Y-%m-%d %H:%M'),
                ended_at=datetime.utcnow().strftime('%Y-%m-%d %H:%M'),

            )

            db.add(new_session)  # Add the session to the database
            db.commit()          # Commit the changes
            db.refresh(new_session)  # Refresh the instance to get the generated data

        except Exception as e:
            db.rollback() 
            raise HTTPException(status_code=500, detail=f"Failed to add session: {str(e)}")


        return {
            "id": user_id,
            "role": user_role,
            "token": token,
            "message": "Session successfully created.",
            "session_id": new_session.session_id,
        }


