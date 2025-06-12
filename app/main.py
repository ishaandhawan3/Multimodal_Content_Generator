import streamlit as st
from models.text_generator import TextGenerator
from models.image_generator import ImageGenerator
from models.video_generator import VideoGenerator
from models.audio_generator import AudioGenerator

st.title("Multimodal Content Generator")

modality = st.selectbox("Choose content type", ["Text", "Image", "Video", "Audio"])
prompt = st.text_area("Enter your prompt")

if st.button("Generate"):
    if modality == "Text":
        generator = TextGenerator()
        result = generator.generate(prompt)
        st.write(result)
    elif modality == "Image":
        generator = ImageGenerator()
        image = generator.generate(prompt)
        st.image(image)
    elif modality == "Video":
        generator = VideoGenerator()
        video_frames = generator.generate(prompt)
        # For simplicity, display first frame
        st.image(video_frames[0])
    elif modality == "Audio":
        generator = AudioGenerator()
        audio = generator.generate_speech(prompt)
        st.audio(audio.cpu().numpy(), sample_rate=22050)
