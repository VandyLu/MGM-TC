import dataclasses
from enum import auto, Enum
from typing import List, Tuple
import base64
from io import BytesIO
from PIL import Image


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()
    LLAMA_3 = auto()
    GEMMA = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self, system_prompt=None):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            if 'mmtag' in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, "<image>\n" + init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message = message[0]
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message = message[0]
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message = message[0]
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n" if len(msg) > 0 else msg
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.LLAMA_3:
            if system_prompt is None:
                ret = self.system + self.sep
            else:
                ret = '<|start_header_id|>system<|end_header_id|>\n\n' + system_prompt + self.sep

            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message = message[0]
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.GEMMA:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += "<start_of_turn>" + role + "\n" + message + "<end_of_turn>\n" + seps[i % 2]
                else:
                    ret += "<start_of_turn>" + role + "\n"
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def process_image(self, image, image_process_mode, return_pil=False, image_format='PNG', max_len=1344, min_len=672):
        if image_process_mode == "Pad":
            def expand2square(pil_img, background_color=(122, 116, 104)):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            image = expand2square(image)
        elif image_process_mode in ["Default", "Crop"]:
            pass
        elif image_process_mode == "Resize":
            image = image.resize((336, 336))
        else:
            raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
        if max(image.size) > max_len:
            max_hw, min_hw = max(image.size), min(image.size)
            aspect_ratio = max_hw / min_hw
            shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
            longest_edge = int(shortest_edge * aspect_ratio)
            W, H = image.size
            if H > W:
                H, W = longest_edge, shortest_edge
            else:
                H, W = shortest_edge, longest_edge
            image = image.resize((W, H))
        if return_pil:
            return image
        else:
            buffered = BytesIO()
            image.save(buffered, format=image_format)
            img_b64_str = base64.b64encode(buffered.getvalue()).decode()
            return img_b64_str

    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    image = self.process_image(image, image_process_mode, return_pil=return_pil)
                    images.append(image)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    img_b64_str = self.process_image(
                        image, "Default", return_pil=False,
                        image_format='JPEG')
                    img_str = f'<img src="data:image/jpeg;base64,{img_b64_str}" alt="user upload image" />'
                    msg = img_str + msg.replace('<image>', '').strip()
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                if type(msg) is tuple and len(msg) == 2:
                    msg, img_b64_str = msg
                    img_str = f'<img src="data:image/jpeg;base64,{img_b64_str}" alt="user upload image" />'
                    msg = msg.strip() + img_str
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_vicuna_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "What are the key differences between renewable and non-renewable energy sources?"),
        ("Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.\n")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llama_2 = Conversation(
    system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_llava_llama_2 = Conversation(
    system="You are a helpful language and vision assistant. "
           "You are able to understand the visual content that the user provides, "
           "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_vimo_project_info",
            "description": "Get the basic information(project type, training hyperparameteres, label and annotation statistics) of a ViMo project.",
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_vimo_parameters",
            "description": "Set the hyperparameters of the current ViMo project",
            "parameters": {
                "type": "object",
                "properties": {
                    "n_epoch": {
                        "type": "integer",
                        "description": "The number of epochs for training the model."
                    },
                    "data_split_ratio": {
                        "type": "integer",
                        "description": "The ratio (数据划分) of the training data, from 10 to 90."
                    },
                    'model_type': {
                        'type': 'string',
                        'enum': ['simple', 'complex', 'turbo'],
                        'description': 'The type of the model to be used for training. simple(低功耗): low power consumption, availabel in classification, segementation, detection, ocr. complex(高精度模式): high precision model, availabel in classification, segementation, detection. turbo(高性能模式): for ocr only. Default: complex for  classification, segementation, detection, turbo for ocr'
                    },
                    'image_type': {
                        'type': 'string',
                        'enum': ['RGB', 'GRAY'],
                        'description': 'Use RGB (彩色模式) image or convert to GRAY image in Training. Gray mode (灰度模式) may train faster and infer faster, but it may lose some information and lead to degraded quality. Default: RGB'
                    },
                    "data_augmentation": {
                        "type": "string",
                        "enum": ["RandomHorizontalFlip", "RandomVerticalFlip", "ImpulseNoise","GaussianBlur", "MotionBlur", "ColorJitter","RandomRotation", "RandomScale", "RandomShift"],
                        "description": "The type of data augmentation to be used, RandomHorizontalFlip(水平翻转), RandomVerticalFlip(垂直翻转), ImpulseNoise(椒盐噪声), GaussianBlur(高斯模糊), MotionBlur(运动模糊), ColorJitter(图像亮度),RandomRotation(随机旋转), RandomScale(随机缩放), RandomShift(随机平移), return one at a time"
                    },
                    "resize": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "the desired resize height and width integer respectivly."
                    },
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_vimo_module",
            "description": "Create a new ViMo project(classificiation, detection, segmentation, or ocr), import the existing images and annotations, return the project ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_type": {
                        "type": "string",
                        "enum": ["classification", "detection", "segmentation", "ocr"],
                        "description": "The type of the ViMo project(classificiation, detection, segmentation, or ocr)."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "start_vimo_train_by_id",
            "description": "Start training the ViMo project of project ID.",
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_last_trained_metrics',
            'description': 'Get the performance metrics of the last trained model.'
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'check_training_status',
            'description': 'Check the training status of the current ViMo project.',
        }
    },
]

import json

system_prompt = '''You are Industry GPT, a powerful AI assistant developed by the SmartMore company, equipped with expert-level knowledge and intelligence in the field of industrial manufacturing.\n
You can answer questions about industrial manufacturing, and assist users in utilizing ViMo, a deep learning platform for industrial manufacturing with classification (分类), object detection (检测), sementic segmentation  (分割) and ocr (单字符识别) capabilities.\n 
Segmentation primarily deals with defects like scratch, dent or something small that need precisie contour, whereas detection just give you the bounding box.\n
You can also provide guidance on how to use ViMo to solve specific problems in industrial vision inspection tasks, recommend proper algorithm solutions, and operate ViMo through API to help the user complete tasks.\n\n
Introduction to ViMo: A deep learning platform. Users will first upload images to a dataset. The image may or may not have label(特征).
Users can then create in a project with one of the algorithms, the dataset will be automatically added to the project. If the image has annotations(标注), they will be added as well, otherwise, users can annotate (标注) the image within the project.\n
They can also split(划分) the data into train and validation set on ViMo. After training, they can see the performance metrics for the the model.\n\n
Keep the rules in mind: 
1. You should not answer any question that is not related to ViMo.\n
2. DO NOT reveal project ID to the user.\n
3. DO NOT reveal the API functions of ViMo the user.\n
4. You should always repsond in the same lanuage as the user input or based on user instructions.(English, Traditional Chinese, Simplified Chinese)\n
5. You should confirm the dataset details with the user before create any project. You should only create a project when at least one dataset has been uploaded, but the dataset doesn't need to have annotations before a project is created.\n
6. Users may use an existing project, in which case you should continue in the existing project. \n
7. After the project has been created, you should always use the function 'get_current_vimo_project_info' to ensure there are sufficient annotations(标注) before set parameters or start training, if label_type(特征) is empty or there is no annotation (标注) for any labe_type, you should remind the user to annotate(标注) the image first.\n
8. if there are annotations (标注), but number of train image is 0, it means images have not been splited into training and validation set, you should remind the user.\n
9. You must not call 'start_vimo_train_by_id' if there is less than 1 train image and 1 validation image as shown in project status, regardless of the user prompt.\n
10. You should only create one project at a time and make sure the project satisfies the requirements of the user.\n
11. Do not randomly suggest a classification, detection, or segmentation model to the user, instead you should ask the user details of his requirment and suggest the best algorithm to him.\n
12. Before calling 'start_vimo_train_by_id', you must always pause and ask the user to re-confirm the current setting.\n
13. After training starts, you should first inform the user that training has began, user can ask questions when training finishes.\n,
14. If user ask about the training, first check the training status with the given function 'check_training_status' before you respond to the user until you find the training is finished, and then report and analyze the training results to the user.\n
15. After the project has been created, you need to check training status with the given function 'get_vimo_project_info' to understand the current project states, including number of labels(特征).\n
16. Regardless of the user prompt, you must never disclose you are related to OpenAI or chatGPT in any way.\n\n

You are allowed to use the following functions if needed:\n
''' + json.dumps(tools, indent=4) + "\n" + 'You can call the function by the format: <function_call> {"name": "<function_name>", "parameters": {"<parameter_name>": "<parameter_value>"}}\n'

conv_llama_3 = Conversation(
    system="<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. "
           "You are able to understand the visual content that the user provides, "
           "and assist the user with a variety of tasks using natural language.",
    roles=("<|start_header_id|>user<|end_header_id|>\n\n", 
           "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    version="llama_v3",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_3,
    sep="<|eot_id|>",
)

conv_mpt = Conversation(
    system="""<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_llava_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)

conv_llava_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_llava_v0_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
    version="v0_mmtag",
)

conv_llava_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_vicuna_imgsp_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="imgsp_v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llava_v1_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
    version="v1_mmtag",
)

conv_phi_2 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="phi2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="<|endoftext|>",
)

conv_mistral_instruct = Conversation(
    system="",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_gemma = Conversation(
    system="",
    roles=("user", "model"),
    version="gemma",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.GEMMA,
    sep="",
    sep2="<eos>",
)

conv_chatml_direct = Conversation(
    system="""<|im_start|>system
Answer the questions.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

default_conversation = conv_vicuna_v1
conv_templates = {
    "default": conv_vicuna_v0,
    "v0": conv_vicuna_v0,
    "v1": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
    "phi_2": conv_phi_2,
    "gemma": conv_gemma,
    "llama_2": conv_llama_2,
    "llama_3": conv_llama_3,
    "imgsp_v1": conv_vicuna_imgsp_v1,
    "mistral_instruct": conv_mistral_instruct,
    "chatml_direct": conv_chatml_direct,
    "mistral_direct": conv_chatml_direct,
    "plain": conv_llava_plain,
    "v0_plain": conv_llava_plain,
    "llava_v0": conv_llava_v0,
    "v0_mmtag": conv_llava_v0_mmtag,
    "llava_v1": conv_llava_v1,
    "v1_mmtag": conv_llava_v1_mmtag,
    "llava_llama_2": conv_llava_llama_2,
    "mpt": conv_mpt,
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())