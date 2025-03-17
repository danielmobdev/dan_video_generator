import os
import requests
import google.generativeai as genai
from gtts import gTTS
from moviepy.editor import *
from moviepy.audio.fx import all as audio_fx
from PIL import Image
import re
import time
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
os.environ["FFMPEG_BINARY"] = "ffmpeg"
os.environ["IMAGEMAGICK_BINARY"] = "convert"

# Configuration
config = {
    'GEMINI_API_KEY': 'AIzaSyDYU85enZKoPqAJVF3hlcySstAqhqg-tQE',
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
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        
        if not os.path.exists(config['FONT_PATH']):
            raise FileNotFoundError(f"Font file not found: {config['FONT_PATH']}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_gen = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.image_gen.safety_checker = None
        self.image_gen.enable_attention_slicing()
        self._ensure_background_music()

    def _ensure_background_music(self):
        if not os.path.exists(self.config['BACKGROUND_MUSIC_PATH']):
            logger.info("Downloading background music...")
            try:
                response = requests.get(self.config['BACKGROUND_MUSIC_URL'], stream=True, timeout=10)
                response.raise_for_status()
                with open(self.config['BACKGROUND_MUSIC_PATH'], 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)
                logger.info("Background music downloaded.")
            except requests.RequestException as e:
                logger.error(f"Failed to download background music: {e}")
                raise
        # Verify the music file
        try:
            temp_audio = AudioFileClip(self.config['BACKGROUND_MUSIC_PATH'])
            logger.info(f"Background music duration: {temp_audio.duration} seconds")
            temp_audio.close()
        except Exception as e:
            logger.error(f"Background music file is invalid: {e}")
            raise

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
        hook_prompt = f"Generate an attention-grabbing hook for a {platform} video about '{prompt}'."
        try:
            response = self.model.generate_content(hook_prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Failed to generate hook: {e}")
            raise

    def generate_script(self, prompt, platform, affiliate_link=None):
        logger.info("Generating script...")
        script_prompt = f"""Create a concise {platform} script about '{prompt}' with 3 short scenes and visual keywords."""
        try:
            response = self.model.generate_content(script_prompt)
            raw_text = response.text.strip()

            pattern = re.compile(r'\*\*Scene (\d+):\*\*\s*\nNarration:\s*(.+?)\nKeywords:\s*(.+?)(?=\n\*\*Scene|\Z)', re.DOTALL)
            scenes = [{'number': m.group(1), 'narration': m.group(2).strip(), 'keywords': m.group(3).strip()} 
                     for m in pattern.finditer(raw_text)]
            
            if affiliate_link:
                scenes.append({
                    'number': '4',
                    'narration': f"Check out the link below: {affiliate_link}",
                    'keywords': 'product showcase, call to action'
                })
            return scenes
        except Exception as e:
            logger.error(f"Failed to generate script: {e}")
            raise

    async def create_custom_video(self, prompt, platform, image_file, affiliate_link=None, output_file="custom_video.mp4"):
        logger.info("Starting video creation...")
        image_path = None
        audio_files = []
        try:
            image_path = f"uploaded_{int(time.time())}.jpg"
            with open(image_path, "wb") as f:
                f.write(await image_file.read())

            img = Image.open(image_path).resize(self.config['VIDEO_SIZE'], Image.Resampling.LANCZOS)
            img.save(image_path)

            hook = self.generate_viral_hook(prompt, platform)
            scenes = self.generate_script(prompt, platform, affiliate_link)
            scenes.insert(0, {'number': '0', 'narration': hook, 'keywords': 'bold, engaging, dynamic'})

            clips = []
            for idx, scene in enumerate(scenes):
                audio_path = f"scene_{idx}.mp3"
                audio_files.append(audio_path)
                tts = gTTS(text=scene['narration'], lang='en', slow=False)
                tts.save(audio_path)
                audio = AudioFileClip(audio_path).fx(audio_fx.volumex, 1.5)
                txt_clip = self._create_text_clip(scene['narration'], audio.duration)
                img_clip = ImageClip(image_path).set_duration(audio.duration)
                composite = CompositeVideoClip([img_clip, txt_clip]).set_audio(audio)
                clips.append(composite.crossfadein(self.config['TRANSITION_DURATION']))

            final_video = concatenate_videoclips(clips, method="compose")
            logger.info(f"Final video duration: {final_video.duration} seconds")

            # Load background music
            bg_music = AudioFileClip(self.config['BACKGROUND_MUSIC_PATH']).fx(audio_fx.volumex, 0.2)
            logger.info(f"Original background music duration: {bg_music.duration} seconds")

            # If background music is shorter, concatenate it manually
            if bg_music.duration < final_video.duration:
                logger.info("Background music is shorter than video, concatenating...")
                loops_needed = int(final_video.duration / bg_music.duration) + 1
                bg_music_list = [bg_music] * loops_needed
                bg_music = concatenate_audioclips(bg_music_list)
                logger.info(f"Concatenated background music duration: {bg_music.duration} seconds")
            
            # Trim to match video duration
            bg_music = bg_music.set_duration(final_video.duration)
            logger.info(f"Trimmed background music duration: {bg_music.duration} seconds")

            final_video.audio = CompositeAudioClip([final_video.audio, bg_music])
            final_video.write_videofile(output_file, fps=24, codec="libx264", audio_codec="aac", threads=4)

            return output_file
        except Exception as e:
            logger.error(f"Video creation failed: {e}")
            raise
        finally:
            # Cleanup
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
            for audio_path in audio_files:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            gc.collect()

creator = YouTubeVideoCreator(config)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate-custom", response_class=FileResponse)
async def generate_custom_video(
    platform: str = Form(...),
    custom_prompt: str = Form(...),
    affiliate_link: str = Form(None),
    images: UploadFile = File(...)
):
    output_file = f"custom_video_{int(time.time())}.mp4"
    try:
        await creator.create_custom_video(custom_prompt, platform, images, affiliate_link, output_file)
        return FileResponse(output_file, media_type="video/mp4", filename="custom_video.mp4")
    except Exception as e:
        logger.error(f"Video generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)