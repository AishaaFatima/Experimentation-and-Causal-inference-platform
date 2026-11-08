from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routers import ab_tests, bayes, causal, data, power, sequential

app = FastAPI(title="Experimentation & Causal Inference Platform", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError):
    """Services raise ValueError for bad inputs; return them as readable 422s."""
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(ModuleNotFoundError)
async def missing_dependency_handler(_: Request, exc: ModuleNotFoundError):
    """Heavy optional libraries (DoWhy, EconML, PyMC) are imported lazily; explain how to install them."""
    return JSONResponse(status_code=501, content={"detail": str(exc)})


for r in (data.router, power.router, ab_tests.router, sequential.router, causal.router, bayes.router):
    app.include_router(r)


@app.get("/health")
def health():
    return {"status": "ok"}
