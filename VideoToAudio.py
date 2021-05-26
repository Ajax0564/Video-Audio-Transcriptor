
# !pip install moviepy
import moviepy.editor as mp

clip = mp.VideoFileClip('../input/video-file/video.mp4')
end = 13
end = min(clip.duration,end)

# Save the paths for later
clip_paths = []

# Extract Audio-only from mp4
for i in range(0, int(end), 10):
  sub_end = min(i+10, end)
  sub_clip = clip.subclip(i,sub_end)

  sub_clip.audio.write_audiofile("audio_" + str(i) + ".mp3")
  clip_paths.append("audio_" + str(i) + ".mp3")