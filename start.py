# start.py
import os
import uvicorn

if __name__ == "__main__":
    # Railway will set PORT in the environment, default to 8000 locally
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
