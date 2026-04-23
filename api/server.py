import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title = "C++ Transformer API", version="1.0")

class GenerateRequest(BaseModel):
    prompt : str
    max_tokens : int = 100


@app.post("/generate")
def generate_text(request : GenerateRequest):
    try:
        #routing to the inference 
        result = subprocess.run(
            ["./gpt_engine", "infer", request.prompt],
            capture_output = True,
            text = True,
            check = True,
        )

        generated_text = result.stdout.strip()

        return {
            "status" : "success",
            "prompt" : request.prompt,
            "output" : generated_text,
        }
    
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Engine crashed: {e.stderr}")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="gpt_engine binary not found. Did you compile it?")
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

