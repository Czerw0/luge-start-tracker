from fastapi import FastAPI

app = FastAPI(title="Luge Start Tracker")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
