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
    },
    'AFFILIATE_TEXT_STYLE': {
        'font_size': 80,
        'text_color': '#00D4FF',
        'stroke_color': '#FFFFFF',
        'stroke_width': 2,
        'bg_color': 'rgba(0, 0, 0, 0.7)',
        'position': ('center', 'bottom')
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
            logger.info("Downloading default background music...")
            try:
                response = requests.get(self.config['BACKGROUND_MUSIC_URL'], stream=True, timeout=10)
                response.raise_for_status()
                with open(self.config['BACKGROUND_MUSIC_PATH'], 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)
                logger.info("Default background music downloaded.")
            except requests.RequestException as e:
                logger.error(f"Failed to download background music: {e}")
                raise
        try:
            temp_audio = AudioFileClip(self.config['BACKGROUND_MUSIC_PATH'])
            logger.info(f"Default background music duration: {temp_audio.duration} seconds")
            temp_audio.close()
        except Exception as e:
            logger.error(f"Default background music file is invalid: {e}")
            raise

    def _create_text_clip(self, text, duration, is_affiliate=False):
        style = self.config['AFFILIATE_TEXT_STYLE'] if is_affiliate else self.config['TEXT_STYLE']
        return TextClip(
            text,
            fontsize=style['font_size'],
            font=self.config['FONT_PATH'],
            color=style['text_color'],
            bg_color=style['bg_color'],
            stroke_color=style['stroke_color'],
            stroke_width=style['stroke_width'],
            size=(self.config['VIDEO_SIZE'][0] - 200, None),
            method='caption',
            align='center'
        ).set_duration(duration).set_position(style['position'])

    def generate_viral_hooks(self, prompt, platform):
        logger.info("Generating viral hooks...")
        hook_prompt = f"""Generate 5 attention-grabbing hooks for a {platform} video about '{prompt}'. Provide them as plain text without brackets, emojis, or special characters. Use only letters, numbers, and basic punctuation (periods, commas, question marks, exclamation marks). Provide them as a list, one per line, in this exact format:
<hook text 1>
<hook text 2>
<hook text 3>
<hook text 4>
<hook text 5>"""
        try:
            response = self.model.generate_content(hook_prompt)
            raw_text = response.text.strip()
            logger.info(f"Raw hooks response: {raw_text}")
            hooks = [line.strip() for line in raw_text.split('\n') if line.strip()]
            if len(hooks) != 5:
                raise ValueError(f"Expected 5 hooks, got {len(hooks)}")
            return hooks
        except Exception as e:
            logger.error(f"Failed to generate hooks: {e}")
            raise

    async def create_custom_video(self, prompt, platform, image_files: list[UploadFile], affiliate_link=None, bg_music_file: UploadFile = None, output_file="custom_video.mp4"):
        logger.info("Starting video creation...")
        image_paths = []
        audio_files = []
        bg_music_path = self.config['BACKGROUND_MUSIC_PATH']
        custom_bg_path = None
        try:
            # Process uploaded images
            for idx, img_file in enumerate(image_files):
                image_path = f"uploaded_{int(time.time())}_{idx}.jpg"
                with open(image_path, "wb") as f:
                    f.write(await img_file.read())
                img = Image.open(image_path).resize(self.config['VIDEO_SIZE'], Image.Resampling.LANCZOS)
                img.save(image_path)
                image_paths.append(image_path)
            logger.info(f"Processed {len(image_paths)} images")

            # Process custom background music if provided
            if bg_music_file:
                custom_bg_path = f"custom_bg_{int(time.time())}.mp3"
                with open(custom_bg_path, "wb") as f:
                    f.write(await bg_music_file.read())
                try:
                    temp_audio = AudioFileClip(custom_bg_path)
                    if temp_audio.duration <= 0:
                        raise ValueError("Audio duration is zero or invalid")
                    logger.info(f"Custom background music duration: {temp_audio.duration} seconds")
                    temp_audio.close()
                    bg_music_path = custom_bg_path
                    logger.info("Using custom background music")
                except Exception as e:
                    logger.warning(f"Invalid custom background music file: {e}. Falling back to default.")
                    bg_music_path = self.config['BACKGROUND_MUSIC_PATH']
            else:
                logger.info("Using default background music")

            # Generate 5 hooks
            hooks = self.generate_viral_hooks(prompt, platform)
            scenes = [{'number': str(i + 1), 'narration': hook, 'keywords': 'bold, engaging, dynamic'} 
                     for i, hook in enumerate(hooks)]
            if affiliate_link:
                scenes.append({
                    'number': str(len(scenes) + 1),
                    'narration': f"Want to know more? Check out this link: {affiliate_link}",
                    'keywords': 'call to action, affiliate link'
                })
            logger.info(f"Total scenes (hooks + affiliate): {len(scenes)}")

            # Create video clips with perfect sync and smooth transitions
            clips = []
            for idx, scene in enumerate(scenes):
                # Generate narration audio
                audio_path = f"scene_{idx}.mp3"
                audio_files.append(audio_path)
                tts = gTTS(text=scene['narration'], lang='en', slow=False)
                tts.save(audio_path)
                audio = AudioFileClip(audio_path).fx(audio_fx.volumex, 1.5)
                
                # Create text and image clips with exact audio duration
                is_affiliate = idx == len(scenes) - 1 and affiliate_link is not None
                txt_clip = self._create_text_clip(scene['narration'], audio.duration, is_affiliate)
                img_path = image_paths[idx % len(image_paths)]  # Cycle through images
                img_clip = ImageClip(img_path).set_duration(audio.duration)
                
                # Combine into a composite clip with only the scene's audio
                composite = CompositeVideoClip([img_clip, txt_clip]).set_audio(audio)
                
                # Apply fade-in and fade-out to visuals only, skipping start and end
                if idx > 0 and idx < len(scenes) - 1:  # Middle slides only
                    composite = composite.fadein(self.config['TRANSITION_DURATION']).fadeout(self.config['TRANSITION_DURATION'])
                elif idx == 0 and len(scenes) > 1:  # First slide with more to follow
                    composite = composite.fadeout(self.config['TRANSITION_DURATION'])
                elif idx == len(scenes) - 1 and idx > 0:  # Last slide with previous slides
                    composite = composite.fadein(self.config['TRANSITION_DURATION'])
                
                clips.append(composite)
                logger.info(f"Created slide {idx + 1}/{len(scenes)}: {scene['narration']} (duration: {audio.duration}s)")

            # Concatenate clips sequentially without overlap
            final_video = concatenate_videoclips(clips, method="chain", padding=0)
            logger.info(f"Final video duration: {final_video.duration} seconds")

            # Handle background music (no narration, just instrumental)
            bg_music = AudioFileClip(bg_music_path).fx(audio_fx.volumex, 0.2)
            logger.info(f"Background music duration: {bg_music.duration} seconds")
            if bg_music.duration < final_video.duration:
                logger.info("Background music is shorter than video, looping...")
                loops_needed = int(final_video.duration / bg_music.duration) + 1
                bg_music = concatenate_audioclips([bg_music] * loops_needed)
                logger.info(f"Looped background music duration: {bg_music.duration} seconds")
            bg_music = bg_music.set_duration(final_video.duration)
            
            # Combine narration and background music, ensuring narration takes precedence
            final_audio = CompositeAudioClip([final_video.audio, bg_music])
            final_video = final_video.set_audio(final_audio).set_duration(final_video.duration)

            # Write video file
            final_video.write_videofile(output_file, fps=24, codec="libx264", audio_codec="aac", threads=4, preset="ultrafast")
            return output_file
        except Exception as e:
            logger.error(f"Video creation failed: {e}")
            raise
        finally:
            # Cleanup
            for path in image_paths:
                if os.path.exists(path):
                    os.remove(path)
            for audio_path in audio_files:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            if custom_bg_path and os.path.exists(custom_bg_path):
                os.remove(custom_bg_path)
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
    images: list[UploadFile] = File(...),
    bg_music: UploadFile = File(None)
):
    output_file = f"custom_video_{int(time.time())}.mp4"
    try:
        await creator.create_custom_video(custom_prompt, platform, images, affiliate_link, bg_music, output_file)
        return FileResponse(output_file, media_type="video/mp4", filename="custom_video.mp4")
    except Exception as e:
        logger.error(f"Video generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Video generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)