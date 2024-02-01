
import argparse
import time
import os
import mdtex2html
import gradio as gr
from threading import Thread
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

parser = argparse.ArgumentParser()
parser.add_argument('--device_id', default=0, type=int, help='Which device to run service. Default: 0.')
parser.add_argument('--checkpoint_path', type=str, default=None, help='The path of model checkpoint.')
parser.add_argument('--host', default="0.0.0.0", type=str, help="Which host ip to run the service. Default: 0.0.0.0.")
parser.add_argument('--port', default=8001, type=int, help='Which port to run the service. Default: None.')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)

tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path, device_map="auto", torch_dtype="auto", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained(args.checkpoint_path, trust_remote_code=True)


class Chatbot(gr.Chatbot):
    """Chatbot with overrode postprocess method"""

    def postprocess(self, y):
        """postprocess"""
        if y is None:
            return []
        for i, (message, response) in enumerate(y):
            y[i] = (
                None if message is None else mdtex2html.convert(message),
                None if response is None else mdtex2html.convert(response),
            )
        return y


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def predict(inputs, bot, history, do_sample, top_k, top_p, temperature, max_length, prompt):
    """predict"""
    if inputs == "":
        raise gr.Error("Input cannot be empty!")

    bot.append((parse_text(inputs), ""))

    for output in model.chat_stream(tokenizer, inputs, 
                                    max_token=max_length, history=history, 
                                    do_sample=do_sample, top_k=top_k,
                                    top_p=top_p, temperature=temperature):
        if (inputs, output) in history:
            new_history = history
        else:
            new_history = history + [(inputs, output)]

        bot[-1] = (parse_text(inputs), parse_text(output))

        yield bot, new_history
        
    print("Generate output: %s", output)


def reset_user_input():
    """reset user input"""
    return gr.update(value='')


def reset_state():
    """reset state"""
    return [], []


def set_do_sample_args(do_sample):
    return {top_k_slider: gr.update(visible=do_sample),
            top_p_slider: gr.update(visible=do_sample),
            temp_number: gr.update(visible=do_sample)}


with gr.Blocks() as demo:
    gr.HTML(f"""<h1 align="center">TuringMM Chat</h1>""")
    with gr.Row():
        with gr.Column(scale=10):
            with gr.Group():
                chatbot = gr.Chatbot(label=fr"Chatbot TuringMM")
                user_input = gr.Textbox(show_label=False, placeholder="随便聊些什么吧...", lines=5)
            with gr.Row():
                with gr.Column(scale=6):
                    submit_btn = gr.Button("提交", variant="primary")
                with gr.Column(scale=6):
                    empty_btn = gr.Button("清空上下文")
        with gr.Column(scale=2):
            with gr.Group():
                do_sample_checkbox = gr.Checkbox(value=model.generation_config.do_sample if model.generation_config.do_sample else False,
                                                label="sampling", info="是否采样生成")
                top_k_slider = gr.Slider(value=10, label="top k", maximum=50, minimum=0, step=1,
                                        visible=model.generation_config.do_sample if model.generation_config.do_sample else False,
                                        info="从概率分布中依据概率最大选择k个单词，建议不要过小导致模型能选择的词汇少")
                top_p_slider = gr.Slider(value=0.9, label="top p", maximum=1, minimum=0.01, step=0.01,
                                        visible=model.generation_config.do_sample if model.generation_config.do_sample else False,
                                        info="即从累计概率超过某一个阈值p的词汇中进行采样，所以0.1意味着只考虑由前10%累计概率组成的词汇")
                temp_number = gr.Number(value=0.85, maximum=2, minimum=0.01, step=0.01, label="temperature",
                                        visible=model.generation_config.do_sample if model.generation_config.do_sample else False,
                                        info="越高意味着模型具备更多的可能性。对于更有创造性的应用，可以尝试0.85以上")
                max_len_number = gr.Number(value=512, minimum=0,label="可处理的最大token数")
                
    chat_history = gr.State([])

    submit_btn.click(predict,
                    [user_input, chatbot, chat_history, do_sample_checkbox, top_k_slider, top_p_slider, temp_number,
                    max_len_number],
                    [chatbot, chat_history],
                    show_progress=True)
    submit_btn.click(reset_user_input, [], [user_input])

    empty_btn.click(reset_state, outputs=[chatbot, chat_history], show_progress=True)

    do_sample_checkbox.change(set_do_sample_args, [do_sample_checkbox], [top_k_slider, top_p_slider, temp_number])


if __name__ == "__main__":
    demo.queue().launch(server_name=args.host, server_port=args.port)