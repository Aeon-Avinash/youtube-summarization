from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from transformers import pipeline
import torch
import gradio as gr
import sys

def get_video_id(url):
    """Extracts video ID from YouTube URL"""
    if 'youtube.com' in url:
        # Extracting video ID from URL
        return url.split('v=')[1].split('&')[0]
    elif 'youtu.be' in url:
        # Extracting video ID from shortened URL
        return url.split('/')[-1]
    else:
        raise ValueError("Invalid YouTube URL")

def get_summary(transcript):
    model_name="Falconsai/text_summarization"
    text_summary_pipeline = pipeline("summarization", model=model_name, device=-1, torch_dtype=torch.bfloat16)          
# use device=-1 for CPU and device=0 for GPU (using litserve we could automate it, but without using a wrapper class 'device' has to be manually specified) 
    video_summary = text_summary_pipeline(transcript)
    return video_summary[0]["summary_text"]


def get_transcript_summary(url):
    """Fetches and prints the transcript of a YouTube video given its URL"""
    try:
        video_id = get_video_id(url)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        formatter = TextFormatter()
        formatted_transcript = formatter.format_transcript(transcript)
        transcript_summary = get_summary(formatted_transcript)
        return transcript_summary
    except Exception as e:
        print("An error occurred:", e)

gr.close_all()

# version 0.1
# demo = gr.Interface(fn=summary, inputs="text", outputs="text")
# version 0.2
demo = gr.Interface(
    fn=get_transcript_summary, 
    inputs=[gr.Textbox(label="Youtube Video URL", lines=2)], 
    outputs=[gr.Textbox(label="YouTube Video's Summary", lines=6)], 
    title="GenAI YouTube Video Summarizer", 
    description="This App Summarizes Youtube Videos")
demo.launch()




# if __name__ == '__main__':
#     if len(sys.argv) != 2:
#         print("Usage: python script.py <youtube_video_url>")
#         sys.exit(1)
    
#     youtube_url = sys.argv[1]
#     get_transcript(youtube_url)
