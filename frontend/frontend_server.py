from fastapi import FastAPI
import gradio as gr
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from frontend_block import get_demo

demo = get_demo()
demo.queue()

logger.info("Starting FastAPI app")
app = FastAPI(title="RAG Frontend",
             description="Frontend interface for RAG assessment")

app = gr.mount_gradio_app(app, demo, path="/")

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8091) 