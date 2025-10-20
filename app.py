from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from typing import Optional
from io import BytesIO
import os
import uuid
import asyncio

# =========================
# 🏗 FastAPI App Setup
# =========================
app = FastAPI(title="AMIT AI Studio")

# Serve static files (CSS, JS, images, HTML) from /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# =========================
# 🏠 Home Page Route
# =========================
@app.get("/")
async def home():
    """
    Serve the main HTML page
    """
    return FileResponse("static/index.html")


# =========================
# 🖼 TXT2IMG Endpoint
# =========================
from txt2img_service import generate_txt2img  # Make sure this import exists

@app.post("/txt2img")
async def txt2img_endpoint(style: str = Form(None), custom_prompt: str = Form(None)):
    """
    Generate image from text prompt using Stable Diffusion
    """
    if not custom_prompt or custom_prompt.strip() == "":
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    # Generate image
    image = generate_txt2img(style_choice=style, custom_prompt=custom_prompt)

    # Convert image to streamable PNG
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")


# =========================
# 🖼 IMG2IMG Endpoint
# =========================
from img2img_service import generate_img2img

@app.post("/img2img")
async def img2img_api(file: UploadFile = File(...), prompt: str = Form(...)):
    """
    Generate a new image based on an uploaded image and a prompt
    """
    # Save uploaded file temporarily
    input_path = f"temp_{uuid.uuid4().hex}.png"
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Run img2img generation
    output_image = generate_img2img(input_image_path=input_path, user_prompt=prompt)

    # Save output image
    output_path = f"generated_{uuid.uuid4().hex}.png"
    output_image.save(output_path)

    # Cleanup input image
    os.remove(input_path)

    return FileResponse(output_path, media_type="image/png", filename=output_path)


# =========================
# 🎭 Shakespeare Text Styler Endpoints
# =========================
from shakespeare_rag import ShakespeareStyler

styler = ShakespeareStyler()

@app.get("/textstyler_full")
async def textstyler_full(user_text: str):
    """
    Return full Shakespearean-styled text for user input
    """
    output = "".join(styler.style_text_stream(user_text))
    return {"output": output}


@app.get("/textstyler_stream")
async def textstyler_stream(user_text: str):
    """
    Stream Shakespearean-styled text token by token via SSE
    """
    async def event_generator():
        for token in styler.style_text_stream(user_text):
            yield f"data: {token}\n\n"  # SSE format
            await asyncio.sleep(0.01)   # slight pause for smooth streaming
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# =========================
# ✅ Optional Health Check
# =========================
@app.get("/health")
async def health():
    """
    Simple health check endpoint
    """
    return {"status": "running", "app": "AMIT AI Studio"}
