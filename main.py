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
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

# Configure FFmpeg and ImageMagick paths (adjust based on your system)
# Near the top, replace the existing change_settings block
change_settings({
    "FFMPEG_BINARY": "ffmpeg",  # Render provides this
    "IMAGEMAGICK_BINARY": "convert"  # Render provides this
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

# FastAPI app instance
app = FastAPI()

# Request model for API input
class VideoRequest(BaseModel):
    category: str
    topic: str
    output_file: Optional[str] = "youtube_short.mp4"

def format_time(seconds):
    """Convert seconds to a human-readable format."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.2f}s" if minutes > 0 else f"{secs:.2f}s"

class YouTubeVideoCreator:
    def __init__(self, config):
        self.config = config
        genai.configure(api_key=config["GEMINI_API_KEY"])
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        self.scenes = []
        
        if not os.path.exists(config['FONT_PATH']):
            raise FileNotFoundError(f"Font file not found: {config['FONT_PATH']}")

        print("⏳ Loading Stable Diffusion model...")
        start_time = time.time()
        self.image_gen = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        self.image_gen = self.image_gen.to("mps" if torch.backends.mps.is_available() else "cpu")
        self.image_gen.safety_checker = None
        self.image_gen.enable_attention_slicing()
        print(f"✅ Model loaded in {format_time(time.time() - start_time)}")

        self._ensure_background_music()

    def _ensure_background_music(self):
        if not os.path.exists(self.config['BACKGROUND_MUSIC_PATH']):
            print("🎵 Downloading background music...")
            start_time = time.time()
            try:
                response = requests.get(self.config['BACKGROUND_MUSIC_URL'], stream=True)
                response.raise_for_status()
                with open(self.config['BACKGROUND_MUSIC_PATH'], 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)
                print(f"✅ Background music downloaded in {format_time(time.time() - start_time)}")
            except Exception as e:
                print(f"⚠️ Failed to download background music: {e}")

    def _craft_dynamic_prompt(self, category, topic, keywords):
        """Dynamically craft a Stable Diffusion prompt with strict no-human rules."""
        category = category.lower()
        topic_lower = topic.lower()
        base_negative_prompt = "humans, people, faces, portraits, crowds, cartoon, anime, blurry, low quality, CGI, text, unrealistic, abstract, painting, sketch, distorted"

        if category == "photorealistic":
            base_prompt = "hyper-realistic 8K shot, {keywords}, vivid details, cinematic lighting"
            style_suffix = "sharp focus, immersive depth, HDR, ultra-detailed textures, lifelike composition, realistic shadows"
        elif category == "stylized":
            base_prompt = "stylized render, {keywords}, bold colors, artistic flair"
            style_suffix = "vibrant energy, soft glow, expressive brushstrokes, dynamic movement, painterly aesthetic"
        elif category == "design":
            base_prompt = "sleek design shot, {keywords}, clean lines, modern vibe"
            style_suffix = "crisp finish, reflective glow, high-end aesthetic, smooth minimalism, professional rendering"
        elif category == "general (artistic)":
            base_prompt = "artistic vision, {keywords}, vibrant hues, textured feel"
            style_suffix = "ethereal vibe, surreal touch, painterly textures, dreamlike atmosphere, abstract essence"
        elif category == "cyberpunk":
            base_prompt = "futuristic cyberpunk scene, {keywords}, neon lighting, high-tech dystopian setting"
            style_suffix = "vibrant neon glow, dark atmosphere, ultra-detailed, holographic reflections, cinematic sci-fi composition"
        elif category == "fantasy":
            base_prompt = "epic fantasy setting, {keywords}, magical aura, mythological influence"
            style_suffix = "dramatic lighting, ethereal glow, highly detailed, immersive depth, legendary atmosphere"
        elif category == "horror":
            base_prompt = "dark horror environment, {keywords}, eerie shadows, terrifying presence"
            style_suffix = "moody atmosphere, deep contrasts, haunting realism, unsettling textures, cinematic tension"
        elif category == "anime":
            base_prompt = "anime-style artwork, {keywords}, cel-shaded details, vibrant colors"
            style_suffix = "expressive character design, smooth lines, dynamic action, detailed background, Ghibli-style aesthetic"
        elif category == "science fiction":
            base_prompt = "sci-fi futuristic scene, {keywords}, advanced technology, cosmic exploration"
            style_suffix = "dystopian setting, highly detailed, cinematic 4K render, otherworldly atmosphere, intricate sci-fi designs"
        elif category == "steampunk":
            base_prompt = "steampunk-inspired artwork, {keywords}, Victorian-era technology, brass and gears"
            style_suffix = "mechanical details, aged textures, historical fantasy, intricate craftsmanship, warm brass tones"
        elif category == "pixel art":
            base_prompt = "pixel art scene, {keywords}, 8-bit aesthetics, retro game vibes"
            style_suffix = "low-resolution charm, nostalgic color palette, crisp pixel edges, classic 16-bit shading"
        else:
            base_prompt = "artistic vision, {keywords}, vibrant hues, textured feel"
            style_suffix = "ethereal vibe, surreal touch, painterly textures, dreamlike atmosphere, abstract essence"

        if "spiritual" in topic_lower:
            context = "sacred serene scene, glowing aura, ancient mystical elements, tranquil atmosphere"
            negative_prompt = f"{base_negative_prompt}, chaotic, urban"
        elif "nature" in topic_lower or "ocean" in topic_lower:
            context = "pristine natural landscape, vivid realistic colors, golden hour light"
            negative_prompt = f"{base_negative_prompt}, urban, artificial elements"
        elif "technology" in topic_lower or "futuristic" in topic_lower:
            context = "cutting-edge technology, sleek metallic surfaces, futuristic design"
            negative_prompt = f"{base_negative_prompt}, rustic, old-fashioned"
        elif "wildlife" in topic_lower:
            context = "majestic wildlife setting, golden tones, intricate natural details"
            negative_prompt = f"{base_negative_prompt}, urban, artificial, humans, faces"
        elif "design" in topic_lower or "architecture" in topic_lower:
            context = "modern architectural elements, elegant structure, reflective textures"
            negative_prompt = f"{base_negative_prompt}, cluttered, outdated"
        elif "food" in topic_lower:
            context = "gourmet presentation, glistening textures, appetizing vivid colors"
            negative_prompt = f"{base_negative_prompt}, unappetizing"
        else:
            context = f"scene relevant to {topic}, detailed and immersive"
            negative_prompt = base_negative_prompt

        prompt = f"{base_prompt}, {context}, {style_suffix}, no humans, no faces"
        return prompt.format(keywords=keywords), negative_prompt

    def _get_image(self, category, topic, keywords):
        """Generate an image with exact 9:16 ratio and no human faces."""
        try:
            print(f"🎨 Generating image for category: {category}, topic: {topic}, keywords: {keywords}")
            start_time = time.time()
            prompt, negative_prompt = self._craft_dynamic_prompt(category, topic, keywords)
            
            image = self.image_gen(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=960,
                width=544,
                num_inference_steps=20,  # Reduced for faster local testing
                guidance_scale=9.0,
                eta=0.8,
                generator=torch.Generator(device="mps" if torch.backends.mps.is_available() else "cpu").manual_seed(random.randint(0, 2**32))
            ).images[0]

            image = image.resize(self.config['VIDEO_SIZE'], Image.Resampling.LANCZOS)
            image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=3))
            image = ImageEnhance.Contrast(image).enhance(1.2)
            image = ImageEnhance.Sharpness(image).enhance(1.5)
            
            image_path = f"generated_{int(time.time())}.jpg"
            image.save(image_path, quality=98, optimize=True)
            print(f"✅ Image generated in {format_time(time.time() - start_time)}")
            return image_path
        except Exception as e:
            print(f"⚠️ Image generation failed: {e}")
            return None

    def _create_text_clip(self, text, duration):
        """Create static text clip without complex animation."""
        try:
            start_time = time.time()
            txt_clip = TextClip(
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
            
            print(f"✅ Text clip created in {format_time(time.time() - start_time)}")
            return txt_clip
        except Exception as e:
            print(f"⚠️ Text creation failed: {e}")
            return None

    def generate_script(self, topic):
        """Generate a context-aware script for the video with no human references."""
        print(f"📜 Generating script for topic: {topic}")
        start_time = time.time()
        prompt = f"""Create a concise YouTube Shorts script about "{topic}". 
        Generate 3-5 scenes with 1-2 short sentences each and visual keywords.
        Ensure narration is engaging and contextually relevant to "{topic}".
        Strictly exclude any human elements, faces, or people from narration and keywords; focus on objects, concepts, or scenes.
        Format:
        
        **Scene 1:**  
        Narration: [Concise, engaging narration]  
        Keywords: [Non-human image keywords]  

        **Scene 2:**  
        Narration: [Concise, engaging narration]  
        Keywords: [Non-human image keywords]  
        """

        try:
            response = self.model.generate_content(prompt)
            raw_text = response.text.strip()
            pattern = re.compile(r'\*\*Scene (\d+):\*\*\s*\nNarration:\s*(.+?)\nKeywords:\s*(.+?)(?=\n\*\*Scene|\Z)', re.DOTALL)
            scenes = [{
                'number': m.group(1),
                'narration': m.group(2).strip(),
                'keywords': m.group(3).strip()
            } for m in pattern.finditer(raw_text)]
            print(f"✅ Script generated in {format_time(time.time() - start_time)}")
            return scenes
        except Exception as e:
            print(f"⚠️ Error generating script: {e}")
            return []

    def create_video(self, category, topic, output_file="youtube_short.mp4"):
        total_start_time = time.time()
        
        self.scenes = self.generate_script(topic)
        if not self.scenes:
            raise ValueError("Script generation failed")

        clips = []
        for idx, scene in enumerate(self.scenes):
            scene_start_time = time.time()
            try:
                print(f"\n🎬 Processing scene {idx+1}/{len(self.scenes)}")
                print(f"📝 Narration: {scene['narration']}")
                print(f"🔍 Keywords: {scene['keywords']}")

                image_path = self._get_image(category, topic, scene['keywords'])
                if not image_path:
                    raise ValueError("Image generation failed")

                audio_start_time = time.time()
                tts = gTTS(text=scene['narration'], lang='en', slow=False)
                audio_path = f"scene_{idx}.mp3"
                tts.save(audio_path)
                audio = AudioFileClip(audio_path)
                duration = audio.duration
                print(f"✅ Audio generated in {format_time(time.time() - audio_start_time)}")

                txt_clip = self._create_text_clip(scene['narration'], duration)
                if not txt_clip:
                    raise ValueError("Text clip creation failed")

                img_clip = ImageClip(image_path).set_duration(duration).set_start(0)
                composite = CompositeVideoClip([img_clip, txt_clip]).set_audio(audio)
                if idx == 0:
                    clips.append(composite)
                else:
                    clips.append(composite.crossfadein(self.config['TRANSITION_DURATION']))

                print(f"✅ Scene {idx+1} processed in {format_time(time.time() - scene_start_time)}")
            except Exception as e:
                print(f"⚠️ Scene {idx+1} failed: {str(e)}")
                continue

        if clips:
            print("\n🎥 Combining scenes...")
            final_video = concatenate_videoclips(clips, method="compose")
            
            if os.path.exists(self.config['BACKGROUND_MUSIC_PATH']):
                try:
                    print("\n🎶 Adding background music...")
                    bg_music = AudioFileClip(self.config['BACKGROUND_MUSIC_PATH']).volumex(0.2)
                    bg_music = bg_music.fx(audio_loop, duration=final_video.duration)
                    bg_music = audio_fadein(bg_music, 1).fx(audio_fadeout, 1)
                    final_video.audio = CompositeAudioClip([final_video.audio, bg_music])
                except Exception as e:
                    print(f"⚠️ Error adding background music: {e}")

            print("\n📼 Rendering final video...")
            final_video.write_videofile(
                output_file,
                fps=24,
                codec="libx264",
                audio_codec="aac",
                threads=4,
                logger=None
            )
            print(f"✅ Video rendered: {output_file}")
        else:
            raise ValueError("No valid clips created")
        print(f"⏰ Total time taken: {format_time(time.time() - total_start_time)}")

# Instantiate the creator globally
creator = YouTubeVideoCreator(config)

@app.post("/generate-video/", response_class=FileResponse)
async def generate_video(request: VideoRequest):
    try:
        output_file = request.output_file or f"video_{int(time.time())}.mp4"
        creator.create_video(request.category, request.topic, output_file)
        if os.path.exists(output_file):
            return FileResponse(output_file, media_type="video/mp4", filename=output_file)
        raise HTTPException(status_code=500, detail="Video generation failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Video API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)