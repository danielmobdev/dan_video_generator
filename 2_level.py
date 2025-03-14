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

# Configure FFmpeg and ImageMagick paths
change_settings({
    "FFMPEG_BINARY": "/opt/homebrew/bin/ffmpeg",
    "IMAGEMAGICK_BINARY": "/opt/homebrew/bin/convert"
})

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
        """Dynamically craft a Stable Diffusion prompt based on category and topic."""
        category = category.lower()
        topic_lower = topic.lower()
        base_negative_prompt = "cartoon, anime, blurry, low quality, CGI, text, unrealistic, abstract, painting, sketch, distorted, people"

        # Category-specific styles
        if category == "photorealistic":
            base_prompt = "hyper-realistic 8K photograph, {keywords}, photo-quality, lifelike details, professional studio lighting"
            style_suffix = "cinematic lighting, intricate details, sharp focus, high resolution"
        elif category == "stylized":
            base_prompt = "stylized high-detail rendering, {keywords}, vibrant colors, artistic composition"
            style_suffix = "dynamic pose, soft gradients, ethereal lighting"
        elif category == "design":
            base_prompt = "sleek modern design render, {keywords}, clean lines, sophisticated aesthetic"
            style_suffix = "reflective surfaces, crisp details, high-tech finish"
        elif category == "general (artistic)":
            base_prompt = "artistic depiction, {keywords}, vibrant tones, textured elements"
            style_suffix = "ethereal glow, surreal atmosphere, intricate patterns"
        else:
            print(f"⚠️ Unknown category '{category}'. Defaulting to 'Photorealistic'.")
            base_prompt = "hyper-realistic 8K photograph, {keywords}, photo-quality, lifelike details, professional studio lighting"
            style_suffix = "cinematic lighting, intricate details, sharp focus, high resolution"

        # Topic-specific enhancements
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
            negative_prompt = f"{base_negative_prompt}, urban, artificial"
        elif "design" in topic_lower or "architecture" in topic_lower:
            context = "modern architectural elements, elegant structure, reflective textures"
            negative_prompt = f"{base_negative_prompt}, cluttered, outdated"
        elif "food" in topic_lower:
            context = "gourmet presentation, glistening textures, appetizing vivid colors"
            negative_prompt = f"{base_negative_prompt}, unappetizing"
        else:
            context = f"scene relevant to {topic}, detailed and immersive"
            negative_prompt = base_negative_prompt

        # Combine into final prompt
        prompt = f"{base_prompt}, {context}, {style_suffix}, no humans"
        return prompt.format(keywords=keywords), negative_prompt

    def _get_image(self, category, topic, keywords):
        """Generate an image using a dynamically crafted Stable Diffusion prompt."""
        try:
            print(f"🎨 Generating image for category: {category}, topic: {topic}, keywords: {keywords}")
            start_time = time.time()
            prompt, negative_prompt = self._craft_dynamic_prompt(category, topic, keywords)
            
            image = self.image_gen(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=768,
                width=512,
                num_inference_steps=50,
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
        """Generate a context-aware script for the video."""
        print(f"📜 Generating script for topic: {topic}")
        start_time = time.time()
        prompt = f"""Create a concise YouTube Shorts script about "{topic}". 
        Generate 3-5 scenes with 1-2 short sentences each and visual keywords.
        Ensure narration is engaging and contextually relevant to "{topic}".
        Strictly exclude any human elements from keywords; focus on objects, concepts, or scenes that can be depicted realistically.
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
            print("❌ Script generation failed")
            return

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

                composite_start_time = time.time()
                img_clip = ImageClip(image_path).set_duration(duration)
                composite = CompositeVideoClip([img_clip, txt_clip]).set_audio(audio)
                clips.append(composite.crossfadein(self.config['TRANSITION_DURATION']))
                print(f"✅ Composite clip created in {format_time(time.time() - composite_start_time)}")
                print(f"✅ Scene {idx+1} processed in {format_time(time.time() - scene_start_time)}")

            except Exception as e:
                print(f"⚠️ Scene {idx+1} failed: {str(e)}")
                continue

        if clips:
            print("\n🎥 Combining scenes...")
            combine_start_time = time.time()
            final_video = concatenate_videoclips(clips, method="compose")
            print(f"✅ Scenes combined in {format_time(time.time() - combine_start_time)}")
            
            if os.path.exists(self.config['BACKGROUND_MUSIC_PATH']):
                try:
                    print("\n🎶 Adding background music...")
                    music_start_time = time.time()
                    bg_music = AudioFileClip(self.config['BACKGROUND_MUSIC_PATH']).volumex(0.2)
                    bg_music = bg_music.fx(audio_loop, duration=final_video.duration)
                    bg_music = audio_fadein(bg_music, 1).fx(audio_fadeout, 1)
                    final_video.audio = CompositeAudioClip([final_video.audio, bg_music])
                    print(f"✅ Background music added in {format_time(time.time() - music_start_time)}")
                except Exception as e:
                    print(f"⚠️ Error adding background music: {e}")

            print("\n📼 Rendering final video...")
            render_start_time = time.time()
            final_video.write_videofile(
                output_file,
                fps=24,
                codec="libx264",
                audio_codec="aac",
                threads=4,
                ffmpeg_params=['-movflags', '+faststart'],
                logger=None
            )
            print(f"✅ Video rendered in {format_time(time.time() - render_start_time)}")
            print(f"\n✅ Successfully created: {output_file}")
            print(f"⏰ Total time taken: {format_time(time.time() - total_start_time)}")
        else:
            print("❌ No valid clips created")
            print(f"⏰ Total time taken: {format_time(time.time() - total_start_time)}")

if __name__ == "__main__":
    print("Available categories: Photorealistic, Stylized, Design, General (Artistic)")
    category = input("Enter the category for Stable Diffusion prompts: ").strip()
    topic = input("Enter YouTube Short topic: ").strip()
    creator = YouTubeVideoCreator(config)
    creator.create_video(category, topic)