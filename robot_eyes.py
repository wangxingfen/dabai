from openai import OpenAI
import json
from PIL import Image
import io
import base64
def process_openai_stream(completion):
    """处理 OpenAI 流式响应"""
    for chunk in completion:
            bit=chunk.choices[0].delta
            yield bit
def convert_image_to_webp_base64(input_image_path):
    try:
        with Image.open(input_image_path) as img:
            byte_arr = io.BytesIO()
            img.save(byte_arr, format='webp')
            byte_arr = byte_arr.getvalue()
            base64_str = base64.b64encode(byte_arr).decode('utf-8')
            return base64_str
    except IOError:
        print(f"Error: Unable to open or convert the image {input_image_path}")
        return None
def ai_respose2(input_image_path):
    '''自动化操作控制电脑'''
    with open("settings.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    client = OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        
    )
    
    completion = client.chat.completions.create(
        model=config['model'],
        stream=True,
        messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{convert_image_to_webp_base64(input_image_path)}"
                    }
                },
                {
                    "type": "text",
                    "text": "Describe the image."
                }
            ]
        }]
    )
    ai_respose=''
    for bite in process_openai_stream(completion):
        if bite.reasoning_content:
            r_content=bite.reasoning_content
            #ai_respose+=str(r_content)
            print(str(r_content),end='')
        elif bite.content:
                ai_respose+=str(bite.content)
                print(str(bite.content),end='')
    print(ai_respose)