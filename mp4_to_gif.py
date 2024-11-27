from moviepy.editor import ImageClip, VideoFileClip, concatenate_videoclips
from pathlib import Path

def convert_video_to_gif(input_file, output_file, fps=10):
    clip = VideoFileClip(input_file)


    num_frames = int(clip.fps * clip.duration) 
    print(f"Total frames: {num_frames}, FPS: {clip.fps}")


    VideoFileClip(input_file) 
    frames = [clip.get_frame(t) for t in range(int(clip.duration))] 
    # 各フレームの持続時間を設定 
    frame_clips = [ImageClip(frame).set_duration(10) for frame in frames] 
    # フレームを連結してGIFを作成 
    gif_clip = concatenate_videoclips(frame_clips, method="compose")


    gif_clip.write_gif(output_file, fps=fps)

    gif = VideoFileClip(output_file)
    print(gif.fps, gif.duration)

if __name__ == '__main__':

    input_path_l = [
        '/mnt/d/results/20241127/output/Mi11Lite_output-3120-00035-2024-11-18T170527_CT_ESTDAN_v3_BSD_3ms24ms_GOPRO.mp4',
        '/mnt/d/results/20241127/output/Mi11Lite_output-3120-00035-2024-11-21T192622_C_ESTDAN_v3_BSD_3ms24ms_GOPRO.mp4',
        '/mnt/d/results/20241127/output/Mi11Lite_output-3120-00035-2024-11-25T193451__F_ESTDAN_v3_.mp4',
        '/mnt/d/results/20241127/output/Mi11Lite_output-4838-00063-2024-11-18T170527_CT_ESTDAN_v3_BSD_3ms24ms_GOPRO.mp4',
        '/mnt/d/results/20241127/output/Mi11Lite_output-4838-00063-2024-11-21T192622_C_ESTDAN_v3_BSD_3ms24ms_GOPRO.mp4',
        '/mnt/d/results/20241127/output/Mi11Lite_output-4838-00063-2024-11-25T193451__F_ESTDAN_v3_.mp4',
        '/mnt/d/results/20241127/output/Mi11Lite_output-5150-00041-2024-11-18T170527_CT_ESTDAN_v3_BSD_3ms24ms_GOPRO.mp4',
        '/mnt/d/results/20241127/output/Mi11Lite_output-5150-00041-2024-11-21T192622_C_ESTDAN_v3_BSD_3ms24ms_GOPRO.mp4',
        '/mnt/d/results/20241127/output/Mi11Lite_output-5150-00041-2024-11-25T193451__F_ESTDAN_v3_.mp4',
    ]

    fps = 1

    for input_path in input_path_l:
        output_path = str(Path(input_path).with_suffix('.gif'))
        print(f'Converting {input_path}')
        convert_video_to_gif(input_path, output_path, fps)
