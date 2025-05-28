# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import gc
import os.path as osp
import os
import sys
import warnings

import gradio as gr

warnings.filterwarnings('ignore')

# Model
sys.path.insert(0, os.path.sep.join(osp.realpath(__file__).split(os.path.sep)[:-2]))
import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video

# Global Var
prompt_expander = None
wan_flf2v_720P = None


# Button Func
def load_model(value):
    global wan_flf2v_720P

    if value == '------':
        print("No model loaded")
        return '------'

    if value == '720P':
        if args.ckpt_dir_720p is None:
            print("Please specify the checkpoint directory for 720P model")
            return '------'
        if wan_flf2v_720P is not None:
            pass
        else:
            gc.collect()

            print("load 14B-720P flf2v model...", end='', flush=True)
            cfg = WAN_CONFIGS['flf2v-14B']
            wan_flf2v_720P = wan.WanFLF2V(
                config=cfg,
                checkpoint_dir=args.ckpt_dir_720p,
                device_id=0,
                rank=0,
                t5_fsdp=False,
                dit_fsdp=False,
                use_usp=False,
            )
            print("done", flush=True)
            return '720P'
    return value


def prompt_enc(prompt, img_first, img_last, tar_lang):
    print('prompt extend...')
    if img_first is None or img_last is None:
        print('Please upload the first and last frames')
        return prompt
    global prompt_expander
    prompt_output = prompt_expander(
        prompt, image=[img_first, img_last], tar_lang=tar_lang.lower())
    if prompt_output.status == False:
        return prompt
    else:
        return prompt_output.prompt


def flf2v_generation(flf2vid_prompt, flf2vid_image_first, flf2vid_image_last, resolution, sd_steps,
                   guide_scale, shift_scale, seed, n_prompt):

    if resolution == '------':
        print(
            'Please specify the resolution ckpt dir or specify the resolution'
        )
        return None

    else:
        if resolution == '720P':
            global wan_flf2v_720P
            video = wan_flf2v_720P.generate(
                flf2vid_prompt,
                flf2vid_image_first,
                flf2vid_image_last,
                max_area=MAX_AREA_CONFIGS['720*1280'],
                shift=shift_scale,
                sampling_steps=sd_steps,
                guide_scale=guide_scale,
                n_prompt=n_prompt,
                seed=seed,
                offload_model=True)
            pass
        else:
            print(
                'Sorry, currently only 720P is supported.'
            )
            return None

        cache_video(
            tensor=video[None],
            save_file="example.mp4",
            fps=16,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))

        return "example.mp4"


# Interface
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("""
                    <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
                        Wan2.1 (FLF2V-14B)
                    </div>
                    <div style="text-align: center; font-size: 16px; font-weight: normal; margin-bottom: 20px;">
                        Wan: Open and Advanced Large-Scale Video Generative Models.
                    </div>
                    """)

        with gr.Row():
            with gr.Column():
                resolution = gr.Dropdown(
                    label='Resolution',
                    choices=['------', '720P'],
                    value='------')
                flf2vid_image_first = gr.Image(
                    type="pil",
                    label="Upload First Frame",
                    elem_id="image_upload",
                )
                flf2vid_image_last = gr.Image(
                    type="pil",
                    label="Upload Last Frame",
                    elem_id="image_upload",
                )
                flf2vid_prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the video you want to generate",
                )
                tar_lang = gr.Radio(
                    choices=["ZH", "EN"],
                    label="Target language of prompt enhance",
                    value="ZH")
                run_p_button = gr.Button(value="Prompt Enhance")

                with gr.Accordion("Advanced Options", open=True):
                    with gr.Row():
                        sd_steps = gr.Slider(
                            label="Diffusion steps",
                            minimum=1,
                            maximum=1000,
                            value=50,
                            step=1)
                        guide_scale = gr.Slider(
                            label="Guide scale",
                            minimum=0,
                            maximum=20,
                            value=5.0,
                            step=1)
                    with gr.Row():
                        shift_scale = gr.Slider(
                            label="Shift scale",
                            minimum=0,
                            maximum=20,
                            value=5.0,
                            step=1)
                        seed = gr.Slider(
                            label="Seed",
                            minimum=-1,
                            maximum=2147483647,
                            step=1,
                            value=-1)
                    n_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="Describe the negative prompt you want to add"
                    )

                run_flf2v_button = gr.Button("Generate Video")

            with gr.Column():
                result_gallery = gr.Video(
                    label='Generated Video', interactive=False, height=600)

        resolution.input(
            fn=load_model, inputs=[resolution], outputs=[resolution])

        run_p_button.click(
            fn=prompt_enc,
            inputs=[flf2vid_prompt, flf2vid_image_first, flf2vid_image_last, tar_lang],
            outputs=[flf2vid_prompt])

        run_flf2v_button.click(
            fn=flf2v_generation,
            inputs=[
                flf2vid_prompt, flf2vid_image_first, flf2vid_image_last, resolution, sd_steps,
                guide_scale, shift_scale, seed, n_prompt
            ],
            outputs=[result_gallery],
        )

    return demo


# Main
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a video from a text prompt or image using Gradio")
    parser.add_argument(
        "--ckpt_dir_720p",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")

    args = parser.parse_args()
    assert args.ckpt_dir_720p is not None, "Please specify the checkpoint directory."

    return args


if __name__ == '__main__':
    args = _parse_args()

    print("Step1: Init prompt_expander...", end='', flush=True)
    if args.prompt_extend_method == "dashscope":
        prompt_expander = DashScopePromptExpander(
            model_name=args.prompt_extend_model, is_vl=True)
    elif args.prompt_extend_method == "local_qwen":
        prompt_expander = QwenPromptExpander(
            model_name=args.prompt_extend_model, is_vl=True, device=0)
    else:
        raise NotImplementedError(
            f"Unsupport prompt_extend_method: {args.prompt_extend_method}")
    print("done", flush=True)

    demo = gradio_interface()
    demo.launch(server_name="0.0.0.0", share=False, server_port=7860)
