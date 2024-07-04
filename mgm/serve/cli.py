import argparse
import torch

from mgm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from mgm.conversation import conv_templates, SeparatorStyle
from mgm.model.builder import load_pretrained_model
from mgm.utils import disable_torch_init
from mgm.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
try:
    from diffusers import StableDiffusionXLPipeline
except:
    print('please install diffusers==0.26.3')

try:
    from paddleocr import PaddleOCR
except:
    print('please install paddleocr following https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/README_en.md')


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

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
            "name": "create_vimo_project",
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

tool_string = "\n\n".join(json.dumps(tool) for tool in tools)
system_prompt = '''You are Industry GPT, a powerful AI assistant developed by the SmartMore company, equipped with expert-level knowledge and intelligence in the field of industrial manufacturing.\n
''' + "You can access to the following functions. Use them if needed:\n" + tool_string + '\n' + '''
You can answer questions about industrial manufacturing, and assist users in utilizing ViMo, a deep learning platform for industrial manufacturing with classification (分类), object detection (检测), sementic segmentation  (分割) and ocr (单字符识别) capabilities.\n 
Segmentation primarily deals with defects like scratch, dent or something small that need precisie contour, whereas detection just give you the bounding box.\n
You can also provide guidance on how to use ViMo to solve specific problems in industrial vision inspection tasks, recommend proper algorithm solutions, and operate ViMo through API to help the user complete tasks.\n\n
Introduction to ViMo: A deep learning platform. Users will first upload images to a dataset. The image may or may not have label(特征).
Users can then create in a project with one of the algorithms, the dataset will be automatically added to the project. If the image has annotations(标注), they will be added as well, otherwise, users can annotate (标注) the image within the project.\n
They can also split(划分) the data into train and validation set on ViMo. After training, they can see the performance metrics for the the model.\n\n
Keep the rules in mind: 

1. DO NOT reveal project ID to the user.\n
2. DO NOT reveal the API functions of ViMo the user.\n
'''


# 1. You should not answer any question that is not related to ViMo.\n
# 2. DO NOT reveal project ID to the user.\n
# 3. DO NOT reveal the API functions of ViMo the user.\n
# 4. You should always repsond in the same lanuage as the user input or based on user instructions.(English, Traditional Chinese, Simplified Chinese)\n
# 5. You should confirm the dataset details with the user before create any project. You should only create a project when at least one dataset has been uploaded, but the dataset doesn't need to have annotations before a project is created.\n
# 6. Users may use an existing project, in which case you should continue in the existing project. \n
# 7. After the project has been created, you should always use the function 'get_current_vimo_project_info' to ensure there are sufficient annotations(标注) before set parameters or start training, if label_type(特征) is empty or there is no annotation (标注) for any labe_type, you should remind the user to annotate(标注) the image first.\n
# 8. if there are annotations (标注), but number of train image is 0, it means images have not been splited into training and validation set, you should remind the user.\n
# 9. You must not call 'start_vimo_train_by_id' if there is less than 1 train image and 1 validation image as shown in project status, regardless of the user prompt.\n
# 10. You should only create one project at a time and make sure the project satisfies the requirements of the user.\n
# 11. Do not randomly suggest a classification, detection, or segmentation model to the user, instead you should ask the user details of his requirment and suggest the best algorithm to him.\n
# 12. Before calling 'start_vimo_train_by_id', you must always pause and ask the user to re-confirm the current setting.\n
# 13. After training starts, you should first inform the user that training has began, user can ask questions when training finishes.\n,
# 14. If user ask about the training, first check the training status with the given function 'check_training_status' before you respond to the user until you find the training is finished, and then report and analyze the training results to the user.\n
# 15. After the project has been created, you need to check training status with the given function 'get_vimo_project_info' to understand the current project states, including number of labels(特征).\n
# 16. Regardless of the user prompt, you must never disclose you are related to OpenAI or chatGPT in any way.\n\n
# '''
# You are allowed to use the following functions if needed:\n
# ''' + json.dumps(tools) + "\n" + 'You can call the function by the format: <function_call> {"name": "<function_name>", "parameters": {"<parameter_name>": "<parameter_value>"}}\n'

# system_prompt = """You are a helpful assistant with access to the following functions. Use them if required -
# {"name": "book_tickets", "description": "Book tickets for a specific event", "parameters": {"type": "object", "properties": {"event_name": {"type": "string", "description": "The name of the event"}, "number_of_tickets": {"type": "integer", "description": "The number of tickets to book"}, "seating_preference": {"type": "string", "description": "The preferred seating option"}}, "required": ["event_name", "number_of_tickets"]}}

# {"name": "find_duplicate_elements", "description": "Find duplicate elements in an array", "parameters": {"type": "object", "properties": {"array": {"type": "array", "items": {"type": "string"}, "description": "The array to check for duplicates"}}, "required": ["array"]}}

# {"name": "get_exchange_rate", "description": "Get the exchange rate between two currencies", "parameters": {"type": "object", "properties": {"base_currency": {"type": "string", "description": "The currency to convert from"}, "target_currency": {"type": "string", "description": "The currency to convert to"}}, "required": ["base_currency", "target_currency"]}}

# {"name": "get_famous_quotes", "description": "Get a collection of famous quotes", "parameters": {"type": "object", "properties": {"category": {"type": "string", "description": "The category of quotes"}, "limit": {"type": "integer", "description": "The maximum number of quotes to retrieve"}}, "required": ["category", "limit"]}}\n"""

def main(args):
    # Model
    disable_torch_init()
    
    if args.ocr and args.image_file is not None:
        ocr = PaddleOCR(use_angle_cls=True, use_gpu=True, lang="ch")
        result = ocr.ocr(args.image_file)   
        str_in_image = ''
        if result[0] is not None:
            result = [res[1][0] for res in result[0] if res[1][1] > 0.1]
            if len(result) > 0:
                str_in_image = ', '.join(result)
                print('OCR Token: ' + str_in_image)
    
    if args.gen:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        ).to("cuda")

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    
    if '8x7b' in model_name.lower():
        conv_mode = "mistral_instruct"
    elif '34b' in model_name.lower():
        conv_mode = "chatml_direct"
    elif '2b' in model_name.lower():
        conv_mode = "gemma"
    else:
        conv_mode = "vicuna_v1"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    if args.image_file is not None:
        images = []
        if ',' in args.image_file:
            images = args.image_file.split(',')
        else:
            images = [args.image_file]
        
        image_convert = []
        for _image in images:
            image_convert.append(load_image(_image))
    
        if hasattr(model.config, 'image_size_aux'):
            if not hasattr(image_processor, 'image_size_raw'):
                image_processor.image_size_raw = image_processor.crop_size.copy()
            image_processor.crop_size['height'] = model.config.image_size_aux
            image_processor.crop_size['width'] = model.config.image_size_aux
            image_processor.size['shortest_edge'] = model.config.image_size_aux
        
        # Similar operation in model_worker.py
        image_tensor = process_images(image_convert, image_processor, model.config)
    
        image_grid = getattr(model.config, 'image_grid', 1)
        if hasattr(model.config, 'image_size_aux'):
            raw_shape = [image_processor.image_size_raw['height'] * image_grid,
                        image_processor.image_size_raw['width'] * image_grid]
            image_tensor_aux = image_tensor 
            image_tensor = torch.nn.functional.interpolate(image_tensor,
                                                        size=raw_shape,
                                                        mode='bilinear',
                                                        align_corners=False)
        else:
            image_tensor_aux = []

        if image_grid >= 2:            
            raw_image = image_tensor.reshape(3, 
                                            image_grid,
                                            image_processor.image_size_raw['height'],
                                            image_grid,
                                            image_processor.image_size_raw['width'])
            raw_image = raw_image.permute(1, 3, 0, 2, 4)
            raw_image = raw_image.reshape(-1, 3,
                                        image_processor.image_size_raw['height'],
                                        image_processor.image_size_raw['width'])
                    
            if getattr(model.config, 'image_global', False):
                global_image = image_tensor
                if len(global_image.shape) == 3:
                    global_image = global_image[None]
                global_image = torch.nn.functional.interpolate(global_image, 
                                                            size=[image_processor.image_size_raw['height'],
                                                                    image_processor.image_size_raw['width']], 
                                                            mode='bilinear', 
                                                            align_corners=False)
                # [image_crops, image_global]
                raw_image = torch.cat([raw_image, global_image], dim=0)
            image_tensor = raw_image.contiguous()
            image_tensor = image_tensor.unsqueeze(0)
    
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            image_tensor_aux = [image.to(model.device, dtype=torch.float16) for image in image_tensor_aux]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            image_tensor_aux = image_tensor_aux.to(model.device, dtype=torch.float16)
    else:
        images = None
        image_tensor = None
        image_tensor_aux = []


    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if args.ocr and len(str_in_image) > 0:
            inp = inp + '\nReference OCR Token: ' + str_in_image + '\n'
        if args.gen:
            inp = inp + ' <GEN>'
        # print(inp, '====')

        if images is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = (DEFAULT_IMAGE_TOKEN + '\n')*len(images) + inp
            conv.append_message(conv.roles[0], inp)
            images = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt(system_prompt=system_prompt)
        print(prompt)
        
        # add image split string
        if prompt.count(DEFAULT_IMAGE_TOKEN) >= 2:
            final_str = ''
            sent_split = prompt.split(DEFAULT_IMAGE_TOKEN)
            for _idx, _sub_sent in enumerate(sent_split):
                if _idx == len(sent_split) - 1:
                    final_str = final_str + _sub_sent
                else:
                    final_str = final_str + _sub_sent + f'Image {_idx+1}:' + DEFAULT_IMAGE_TOKEN
            prompt = final_str
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        terminators = tokenizer.eos_token_id
        if "llama_3" in args.conv_mode:
            terminators = [terminators, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                images_aux=image_tensor_aux if len(image_tensor_aux)>0 else None,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                bos_token_id=tokenizer.bos_token_id,  # Begin of sequence token
                eos_token_id=terminators,  # End of sequence token
                pad_token_id=tokenizer.pad_token_id,  # Pad token
                streamer=streamer,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        conv.messages[-1][-1] = outputs
        
        if args.gen and '<h>' in outputs and '</h>' in outputs:
            common_neg_prompt = "out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"
            prompt = outputs.split("</h>")[-2].split("<h>")[-1]
            output_img = pipe(prompt, negative_prompt=common_neg_prompt).images[0]
            output_img.save(args.output_file)
            print(f'Generate an image, save at {args.output_file}')

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None) # file_0.jpg,file_1.jpg for multi image
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--gen", action="store_true")
    parser.add_argument("--output-file", type=str, default='generate.png')
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)