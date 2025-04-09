from fastapi import FastAPI
from fastapi.responses import JSONResponse
import datetime

# Create the FastAPI app instance
# Vercel will look for this 'app' object by default.
app = FastAPI()

@app.get("/api")
async def api_root():
    """
    Root API endpoint.
    Provides a simple welcome message and the current server time.
    """
    now = datetime.datetime.utcnow().isoformat()
    return JSONResponse(content={
        "message": "Welcome to the Barebone Python API!",
        "status": "ok",
        "timestamp_utc": now
    })

@app.get("/api/hello")
async def hello_endpoint(name: str = "World"):
    """
    A simple endpoint that greets the user.
    Takes an optional 'name' query parameter.
    Example: /api/hello?name=Vercel
    """
    return JSONResponse(content={"greeting": f"Hello, {name}!"})

# You can add more endpoints here following the same pattern
# Example POST endpoint (requires sending JSON data in the request body)
# from pydantic import BaseModel
# class Item(BaseModel):
#     id: int
#     description: str

# @app.post("/api/items")
# async def create_item(item: Item):
#    """ Receives item data and returns it """
#    return JSONResponse(content={"received_item": item.dict()})

# Note: For Vercel, you generally don't run the app with uvicorn directly
# in this file. Vercel handles the serving part based on the 'app' object.
# The uvicorn command is used for local development.