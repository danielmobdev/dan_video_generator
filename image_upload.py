import os
import requests
import google.generativeai as genai
from gtts import gTTS
from moviepy.editor import *
from moviepy.config import change_settings
from moviepy.audio.fx.all import audio_loop, audio_fadein, audio_fadeout
from PIL import Image, ImageFilter, ImageEnhance
import re
import time
import random
import torch
from diffusers import StableDiffusionPipeline
from fastapi import FastAPI, HTTPException, Request, Form, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import gc
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure FFmpeg and ImageMagick paths
change_settings({
    "FFMPEG_BINARY": "ffmpeg",
    "IMAGEMAGICK_BINARY": "convert"
})

# Configuration
config = {
    'GEMINI_API_KEY': 'AIzaSyDYU85enZKoPqAJVF3hlcySstAqhqg-tQE',  # Replace with your actual key
    'VIDEO_SIZE': (1080, 1920),
    'TRANSITION_DURATION': 0.5,
    'AUDIO_FADE_DURATION': 0.3,
    'BACKGROUND_MUSIC_URL': "https://lcklrynomxktworgnhhd.supabase.co/storage/v1/object/public/images/uploads/mp3/1.mp3",
    'BACKGROUND_MUSIC_PATH': "background_music.mp3",
    'FONT_PATH': "fonts/BebasNeue-Regular.ttf",
    'TEXT_STYLE': {
        'font_size': 90,
        'text_color': '#FFD700',
        'stroke_color': '#000000',
        'stroke_width': 3,
        'bg_color': 'rgba(0, 0, 0, 0.7)',
        'position': ('center', 'center')
    }
}

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class YouTubeVideoCreator:
    def __init__(self, config):
        self.config = config
        genai.configure(api_key=config["GEMINI_API_KEY"])
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        self.scenes = []
        
        if not os.path.exists(config['FONT_PATH']):
            raise FileNotFoundError(f"Font file not found: {config['FONT_PATH']}")

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.image_gen = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        self.image_gen = self.image_gen.to(self.device)
        self.image_gen.safety_checker = None
        self.image_gen.enable_attention_slicing()
        self._ensure_background_music()

    def _ensure_background_music(self):
        if not os.path.exists(self.config['BACKGROUND_MUSIC_PATH']):
            logger.info("Downloading background music...")
            response = requests.get(self.config['BACKGROUND_MUSIC_URL'], stream=True)
            response.raise_for_status()
            with open(self.config['BACKGROUND_MUSIC_PATH'], 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            logger.info("Background music downloaded.")

    def _create_text_clip(self, text, duration):
        return TextClip(
            text,
            fontsize=self.config['TEXT_STYLE']['font_size'],
            font=self.config['FONT_PATH'],
            color=self.config['TEXT_STYLE']['text_color'],
            bg_color=self.config['TEXT_STYLE']['bg_color'],
            stroke_color=self.config['TEXT_STYLE']['stroke_color'],
            stroke_width=self.config['TEXT_STYLE']['stroke_width'],
            size=(self.config['VIDEO_SIZE'][0] - 200, None),
            method='caption',
            align='center'
        ).set_duration(duration).set_position(self.config['TEXT_STYLE']['position'])

    def generate_viral_hook(self, prompt, platform):
        logger.info("Generating viral hook...")
        hook_prompt = f"""Generate a short, attention-grabbing hook (1-2 sentences) for a {platform} video about '{prompt}'. 
        Make it engaging, platform-specific, and designed to keep viewers watching past the first 3 seconds."""
        response = self.model.generate_content(hook_prompt)
        logger.info("Viral hook generated.")
        return response.text.strip()

    def generate_script(self, prompt, platform, affiliate_link=None):
        logger.info("Generating script...")
        script_prompt = f"""Create a concise {platform} script about '{prompt}'. 
        Generate 3 scenes with 1-2 short sentences each and descriptive visual keywords (no humans). 
        Include an affiliate call-to-action if a link is provided. 
        Format:
        **Scene 1:** Narration: [Text] Keywords: [Keywords]
        **Scene 2:** Narration: [Text] Keywords: [Keywords]
        **Scene 3:** Narration: [Text] Keywords: [Keywords]"""
        response = self.model.generate_content(script_prompt)
        raw_text = response.text.strip()
        pattern = re.compile(r'\*\*Scene (\d+):\*\*\s*\nNarration:\s*(.+?)\nKeywords:\s*(.+?)(?=\n\*\*Scene|\Z)', re.DOTALL)
        scenes = [{'number': m.group(1), 'narration': m.group(2).strip(), 'keywords': m.group(3).strip()} for m in pattern.finditer(raw_text)]
        
        if affiliate_link:
            scenes.append({
                'number': '4',
                'narration': f"Love this? Check out more with the link below!",
                'keywords': 'product showcase, glowing text, sleek design'
            })
        logger.info("Script generated.")
        return scenes

    async def create_custom_video(self, prompt, platform, image_file: UploadFile, affiliate_link=None, output_file="custom_video.mp4"):
        logger.info("Starting video creation...")
        image_path = f"uploaded_{int(time.time())}.jpg"
        with open(image_path, "wb") as f:
            logger.info("Saving uploaded image...")
            f.write(await image_file.read())
            logger.info("Image saved.")

        logger.info("Resizing image...")
        img = Image.open(image_path).resize(self.config['VIDEO_SIZE'], Image.Resampling.LANCZOS)
        img.save(image_path)
        logger.info("Image resized.")

        hook = self.generate_viral_hook(prompt, platform)
        scenes = self.generate_script(prompt, platform, affiliate_link)
        scenes.insert(0, {'number': '0', 'narration': hook, 'keywords': 'dynamic zoom, bold colors'})

        clips = []
        for idx, scene in enumerate(scenes):
            audio_path = f"scene_{idx}.mp3"
            logger.info(f"Generating audio for scene {idx}...")
            tts = gTTS(text=scene['narration'], lang='en', slow=False)
            tts.save(audio_path)
            audio = AudioFileClip(audio_path)
            duration = audio.duration

            logger.info(f"Creating text clip for scene {idx}...")
            txt_clip = self._create_text_clip(scene['narration'], duration)
            img_clip = ImageClip(image_path).set_duration(duration).set_start(0)
            composite = CompositeVideoClip([img_clip, txt_clip]).set_audio(audio)
            clips.append(composite if idx == 0 else composite.crossfadein(self.config['TRANSITION_DURATION']))

            os.remove(audio_path)
            logger.info(f"Scene {idx} processed.")

        logger.info("Concatenating video clips...")
        final_video = concatenate_videoclips(clips, method="compose", padding=-0.1)
        bg_music = AudioFileClip(self.config['BACKGROUND_MUSIC_PATH']).volumex(0.2)
        bg_music = bg_music.fx(audio_loop, duration=final_video.duration).fx(audio_fadein, 1).fx(audio_fadeout, 1)
        final_video.audio = CompositeAudioClip([final_video.audio, bg_music])

        logger.info("Writing final video file...")
        final_video.write_videofile(output_file, fps=24, codec="libx264", audio_codec="aac", threads=2, preset="fast", logger=None)
        os.remove(image_path)
        logger.info("Video creation completed.")
        gc.collect()

creator = YouTubeVideoCreator(config)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate-custom", response_class=FileResponse)
async def generate_custom_video(
    platform: str = Form(...),
    custom_prompt: str = Form(..., alias="custom-prompt"),
    affiliate_link: str = Form(None, alias="affiliate-link"),
    images: UploadFile = File(...)
):
    try:
        output_file = f"custom_video_{int(time.time())}.mp4"
        await creator.create_custom_video(custom_prompt, platform, images, affiliate_link, output_file)
        if os.path.exists(output_file):
            return FileResponse(output_file, media_type="video/mp4", filename=output_file)
        raise HTTPException(status_code=500, detail="Video generation failed")
    except Exception as e:
        logger.error(f"Error during video generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
