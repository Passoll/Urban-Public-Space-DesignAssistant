import streamlit as st
import sys
from io import StringIO
import os
import json
import pdb
import random
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import PIL
import requests
from diskcache import Cache
from openai import OpenAI
from PIL import Image
from termcolor import colored
import AgentUtil

import autogen
from autogen import Agent, AssistantAgent, ConversableAgent, UserProxyAgent
from autogen.agentchat.contrib.img_utils import _to_pil, get_image_data, gpt4v_formatter, get_pil_image
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from autogen.agentchat.contrib.agent_builder import AgentBuilder


if 'agent_saved_path' not in st.session_state:
    st.session_state['agent_saved_path'] = ""

if 'gpt_config' not in st.session_state:
    st.session_state['gpt4v_config'] = []
    st.session_state['gpt_config'] = []
    st.session_state['dalle_config'] = []

termination_notice = (
    '\n\n注意在对话中，请不要表示任何的感恩情感，只说关键的必要的部分，不需要询问对方你还有其他需求或其他任务'
    '如果一旦发现对话中，自己出现"表示感谢"，或者"不用客气"，"祝一切安好","如果您有其他需求，请随时告诉我"。等没有实际知识性意义的内容，请立刻在内容后加入TERMINATE来终止这场对话，并且以当前对话作为你的last message。'
)

if "sysmessage" not in st.session_state:
    st.session_state['sysmessage'] = []

def setupConfig(apikey, base_url, name):
    os.environ["OPENAI_API_KEY"] = apikey
    os.environ["OPENAI_BASE_URL"] = base_url
    save_config = [{
        "model": "",
        "api_key": "",
        "base_url": ""
    }]

    save_config[0]["model"] = name
    save_config[0]["api_key"] = apikey
    save_config[0]["base_url"] = base_url
    filename = 'OAI_CONFIG_LIST'
    st.session_state['gpt_config'] = [
        {
            "model": "gpt-4",
            "api_key": apikey,
        }
    ]
    st.session_state['dalle_config'] = [
        {
            "model": "dalle",
            "api_key": apikey,
        }
    ]
    st.session_state['gpt4v_config'] = [
        {
            "model": "gpt-4-vision-preview",
            "api_key": apikey
        }
    ]
    # Writing JSON data
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(save_config, f, ensure_ascii=False, indent=4)

def buildUpUser(description, i):
    # if not has_buildupAgent:
    #     st.write("Error build up user,please retry generation")
    #     return
    client = OpenAI()  ## if the url is not based on OpenAI, change the chat method
    des_bundle = []
    des_bundle.append(description)
    preprocess = f"你现在需要帮助我将这段第三人称的，对这位建筑使用者的描述,它可能是一段问卷的问答，可能是一段采访，或者一段描述性语言，你需要将其转换为一段第二人称的角色扮演指南，你需要告诉对方尽可能多的细节，希望对方模拟对方的心理活动和偏好。如果描述性文本没有提到，你可以通过联想自行补充一些偏好和细节。你在描述中不需要加入任何前置的回复，如我将为你提供答案等，仅仅表示第二人称的文本。以下为对此人的描述"
    preprocess = preprocess + description
    completion = client.chat.completions.create(
        model = st.session_state["gpt_config"][0]['model'],
        messages=[
            {"role": "system", "content": preprocess}
        ]
    )
    print("User successfully build")
    sysmessage = completion.choices[0].message.content + termination_notice
    st.session_state['sysmessage'].append(sysmessage)
    new_agent = {
        "name": f"architecture_user{i}",
        "model": "gpt-4",
        "system_message": sysmessage,
        "description": description
    }

    with open(st.session_state['agent_saved_path'], 'r', encoding='utf-8') as file:
        data = json.load(file)
        # Assuming 'agent_configs' is a list within the JSON structure
        data['agent_configs'].append(new_agent)

    # Write the updated data back to the file
    with open(st.session_state['agent_saved_path'], 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

    return f"architecture_user{i}"


# 本段配置用户识图功能
gpt4_llm_config = {"config_list": st.session_state["gpt_config"], "cache_seed": 42}

def dalle_call(client: OpenAI, model: str, prompt: str, size: str, quality: str, n: int) -> str:
    # Function implementation...
    cache = Cache(".cache/")  # Create a cache directory
    key = (model, prompt, size, quality, n)
    if key in cache:
        return cache[key]

    # If not in cache, compute and store the result
    response = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
        n=n,
    )
    image_url = response.data[0].url
    img_data = get_image_data(image_url)
    cache[key] = img_data

    return img_data


def extract_img(agent: Agent) -> PIL.Image:
    last_message = agent.last_message()["content"]

    if isinstance(last_message, str):
        img_data = re.findall("<img (.*)>", last_message)[0]
    elif isinstance(last_message, list):
        # The GPT-4V format, where the content is an array of data
        assert isinstance(last_message[0], dict)
        img_data = last_message[0]["image_url"]["url"]

    pil_img = get_pil_image(img_data)
    return pil_img

class DALLEAgent(ConversableAgent):
    def __init__(self, name, llm_config: dict, **kwargs):
        super().__init__(name, llm_config=llm_config, **kwargs)

        try:
            config_list = llm_config["config_list"]
            api_key = config_list[0]["api_key"]
        except Exception as e:
            print("Unable to fetch API Key, because", e)
            api_key = os.getenv("OPENAI_API_KEY")
        self._dalle_client = OpenAI(api_key=api_key)
        self.register_reply([Agent, None], DALLEAgent.generate_dalle_reply)

    def send(
            self,
            message: Union[Dict, str],
            recipient: Agent,
            request_reply: Optional[bool] = None,
            silent: Optional[bool] = False,
    ):
        # override and always "silent" the send out message;
        # otherwise, the print log would be super long!
        super().send(message, recipient, request_reply, silent=True)

    def generate_dalle_reply(self, messages: Optional[List[Dict]], sender: "Agent", config):
        """Generate a reply using OpenAI DALLE call."""
        client = self._dalle_client if config is None else config
        if client is None:
            return False, None
        if messages is None:
            messages = self._oai_messages[sender]

        prompt = messages[-1]["content"]
        # TODO: integrate with autogen.oai. For instance, with caching for the API call
        img_data = dalle_call(
            client=client,
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",  # TODO: the size should be flexible, deciding landscape, square, or portrait mode.
            quality="standard",
            n=1,
        )

        img_data = _to_pil(img_data)  # Convert to PIL image

        # Return the OpenAI message format
        return True, {"content": [{"type": "image_url", "image_url": {"url": img_data}}]}


def chatWithUser(task_des):
    chatlist = []
    archUserProxyList = []
    archUserList = []
    for i in range(len(st.session_state['sysmessage'])):
        # 输出描述词的user
        architecture_user = ConversableAgent(
            name=f"architecture_user{i + 1}",
            llm_config={"config_list": st.session_state['gpt_config'], "max_tokens": 1000
                        },
            system_message=st.session_state['sysmessage'][i],
        )
        arch_UserProxy = autogen.UserProxyAgent(
            name=f"architecture_UserProxy{i + 1}",
            max_consecutive_auto_reply=0,  # terminate without auto-reply
            human_input_mode="NEVER",
            code_execution_config={
                "use_docker": False
            },
        )
        archUserList.append(architecture_user)
        archUserProxyList.append(arch_UserProxy)

    # 负责问Dalle的UserProxy
    architecture_UserProxy = autogen.UserProxyAgent(
        name="architecture_UserProxy",
        max_consecutive_auto_reply=0,  # terminate without auto-reply
        human_input_mode="NEVER",
        code_execution_config={
            "last_n_messages": 1,
            "use_docker": False
        },
    )

    dalle = DALLEAgent(name="Dalle", llm_config={"config_list": st.session_state["dalle_config"]})

    def ask_dalle(message):
        architecture_UserProxy.initiate_chat(dalle, message=message)
        img = extract_img(dalle)
        # plt.imshow(img)
        # plt.axis("off")  # Turn off axis numbers
        # plt.show()
        st.image(img, caption='Generated Image', use_column_width=True)

        return architecture_UserProxy.last_message()["content"]

    def ask_user(message, num):
        archUserProxyList[int(num) - 1].initiate_chat(archUserList[int(num) - 1], message=message)
        # return the last message received from the planner
        return archUserProxyList[int(num) - 1].last_message(archUserList[int(num) - 1])["content"] + "TERMINATE"

    # Main Assitant Agent
    usernum = len(st.session_state['sysmessage'])
    assistant = autogen.AssistantAgent(
        name="assistant",
        system_message=(
            f'你是一个建筑师的助手，你需要完成两个任务，1.总结建筑师给予的任务，向所有建筑使用者（architecture_user）提出咨询。'
            f'建筑使用者一共有{usernum}个，你需要询问他们对场景的想象并邀请他们做出描述，2.将他们对空间的描述利用技能'
            '（ask_dalle）生成每一个人各自的定制化对应的空间场景图片，注意此处需要填入一系列的Prompt进行对场景描述，即一系列描述性短句，若对方的描述不到位，你可以根据对对方的理解加入提示词。'
            f'最终，总共生成的图片数量需要为{usernum}张。在执行任务的过程中，你需要循序渐进进行，先询问第一位用户然后作图，'
            f'然后再询问第二位用户然后作图，途中不需要询问我的意见，等到所有任务完成后再来通知我。注意{termination_notice}'
        ),
        llm_config={
            "temperature": 0,
            "timeout": 600,
            "cache_seed": 42,
            "config_list": st.session_state['gpt_config'],
            "functions": [
                {
                    "name": "ask_dalle",
                    "description": "此任务主要服务于使用者图片生成，通过调用dalle，根据提示词描述生成图片,",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "这是建筑使用者对图片的描述，尽量保证这个描述有足够多的细节，并且是能够被生图模型清晰理解的提示词。同时你需要加入相关的提示词使得生成的图片必须要人视点，色彩风格要接近真实不要卡通风格，里面要存在人的活动。不要轴测图和平面图，要类似照片的风格。尽可能包含更多的人群。例如一张多汁牛排的照片，烤至五分熟并带有烧烤痕迹，摆放在木质盘子上。牛排周围散布着新鲜的迷迭香枝条和一缕酱汁。背景是一张质朴的木制餐桌。来自左上方的温暖、环境光投射出微妙的阴影，突出了牛排的质地。拍摄工具：单反相机，微距摄影，50mm镜头。",
                            },
                        },
                        "required": ["message"],
                    },
                },
                {
                    "name": "ask_user",
                    "description": "此任务主要用于询问使用者（architecture_user）对于建筑项目的看法，参数包括两个，第一个为提问的问题，第二个为使用者的号码，一次执行只能询问一个使用者",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "这是询问建筑项目中需要向使用者提出的问题，使用者是一个非建筑专业人士，需要用通俗易懂的语言翻译项目相关的内容交给使用者进行分析判断,注意内容中需要包含项目的选址，项目的功能和周边情况等基本信息",
                            },
                            "num": {
                                "type": "string",
                                "description": "这是使用者的编码，从1开始计数，即如果询问第一个使用者则用1作为输入参数，第二个则用2,以此类推",
                            },
                        },
                        "required": ["message", "num"],
                    },
                },
            ],
        },
    )

    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: "content" in x and x["content"] is not None and x["content"].rstrip().endswith(
            "TERMINATE"),
        code_execution_config={
            "last_n_messages": 1,
            "work_dir": "planning",
            "use_docker": False,
        },
        function_map={"ask_dalle": ask_dalle,
                      "ask_user": ask_user},
    )

    # 发起第一次询问
    chat_results = user_proxy.initiate_chat(
        assistant,
        message=task_des
    )


def chatwithsingleUser(index,text):
    with open(st.session_state['agent_saved_path'], 'r', encoding='utf-8') as file:
        data = json.load(file)
        name = data["agent_configs"][index]["name"]
        sysMessage = data["agent_configs"][index]["system_message"]

    image_agent = MultimodalConversableAgent(
        name=name,
        max_consecutive_auto_reply=10,
        system_message= sysMessage,
        llm_config={"config_list": st.session_state["gpt4v_config"], "temperature": 0.2, "max_tokens": 300},
    )

    user_proxy = autogen.UserProxyAgent(
        name="User_proxy",
        system_message="A human admin.",
        human_input_mode="NEVER",  # Try between ALWAYS or NEVER
        max_consecutive_auto_reply=0,
        code_execution_config={
            "use_docker": False
        },
    )

    rs = user_proxy.initiate_chat(
        image_agent,
        message=text
    )


def chatwithMultipleuser(task_des, mytask):
    from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
    agent_list = []
    with open(st.session_state['agent_saved_path'], 'r', encoding='utf-8') as file:
        data = json.load(file)
        for agent_config in data["agent_configs"]:
            agent = MultimodalConversableAgent(
                name=agent_config["name"],
                max_consecutive_auto_reply=5,
                llm_config={"config_list": st.session_state['gpt4v_config'], "temperature": 0.2, "max_tokens": 500},  # Adjust as necessary
                system_message=agent_config["system_message"],
            )
            agent_list.append(agent)

    user_proxy = autogen.UserProxyAgent(
        name="User_proxy",
        # system_message="",
        human_input_mode="NEVER",  # Try between ALWAYS, NEVER, and TERMINATE
        max_consecutive_auto_reply=10,
        code_execution_config={
            "last_n_messages": 1,
            "use_docker": False,
            "work_dir": "groupchat",
        },
        # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    )

    inquiry = f"我需要你们关于我的设计方案给出建议，你们作为专家或者使用者，请从你们的视角或者结合专业知识给出判断，并回答我后文提出的问题。我给出的问题中可能包含图片或者方案，也可能会讨论其他的一些设计内容。如果有提供图片，请你们关注图片中的内容并给出评价。注意不需要评价任务书和设计背景本身，不要给我和建议设计无关的建议如:邀请不同的AI助手，不需要总结其他人的建议，给出你自己的意见和评价。请关注我给你提出的具体问题。这个设计方案的具体设计背景与任务书描述为：{task_des}，我需要提出的问题为：{mytask}"

    agent_list.append(user_proxy)
    groupchat = autogen.GroupChat(agents=agent_list, messages=[], max_round=8)

    # vision_capability = VisionCapability(lmm_config={"config_list": gpt_config, "temperature": 0.5, "max_tokens": 300})
    group_chat_manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt4_llm_config)
    # vision_capability.add_to_agent(group_chat_manager)

    rst = user_proxy.initiate_chat(
        group_chat_manager,
        message=inquiry
    )

# Define a class to capture stdout
class CaptureStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout  # Keep track of the original stdout
        sys.stdout = self._stringio = StringIO()  # Redirect stdout to a StringIO object
        return self

    def __exit__(self, type, value, traceback):
        sys.stdout = self._original_stdout  # Restore the original stdout
        self.value = self._stringio.getvalue()  # Store the captured output

def on_input_change():
    user_input = st.session_state.user_input
    st.session_state.past.append(user_input)
    st.session_state.generated.append("The messages from Bot\nWith new line")

def on_btn_click():
    del st.session_state.past[:]
    del st.session_state.generated[:]

def main():
    st.set_page_config(page_title="Architectect Agent for urban space", page_icon=":rocket:", layout="centered",
                       initial_sidebar_state="expanded")
    st.title(":robot_face:智能体辅助空间设计实验")
    st.caption('本实验为基于AutoGen框架的实验性智能体辅助建筑设计项目')

    # Initialize session state for the n  dynamic list if it doesn't exist
    if 'list_items' not in st.session_state:
        st.session_state['list_items'] = []

    agentnum = 4

    # Section 1 设置与专家部署模块
    st.header("基本模块设置")
    st.subheader("模型设置")
    apikey = st.text_input(":id: api_key", key="apikey")
    base_url = st.text_input(":link: url", key="base_url")
    modelname = st.text_input(":scroll: modelname", key="Model Name")
    config_button = st.button("配置模型", use_container_width=True)
    if config_button:
        try:
            setupConfig(apikey, base_url, modelname)
            print("Success!")
            st.write("Success!")
        except:
            st.write("Error!")
    st.subheader("设计情况")
    # Large text box
    task_description = st.text_area(":scroll: 设计任务书输入", height=150)

    st.header("功能模块")
    st.subheader(" :muscle: 专家部署模块")
    # Button to add items to a list
    max_agentNum = st.text_input("专家最大数量", key="max_num")
    agent_request = st.text_area("专家需求（可留空）", height=100)
    # new_item = st.text_input("Add a new item to the list", key="new_item")

    add_item_button = st.button("专家Agent生成",use_container_width=True)
    if add_item_button:
        st.session_state.list_items.clear()
        st.write("开始生成专家代理")
        llm_config = {"temperature": 0}
        builder = AgentBuilder(config_file_or_env="OAI_CONFIG_LIST")
        building_task = f"现有一个建筑空间项目设计任务书如下：{task_description}。请你总结任务书的关键点，给我生成一系列有效的agents辅助建筑设计，为建筑设计流程提出建议。如果任务书中对具体的需求模糊不清，你可以选择根据你的理解生成该项目提出的解决的问题如预算，材质等，进而生成建筑设计辅助的人工智能专家，注意Agent数量不能超过{max_agentNum}个。另外，不需要生成建筑设计师（Architecture designer）"
        agent_list, agent_configs = builder.build(building_task, llm_config)
        st.session_state['agent_saved_path'] = builder.save()
        has_buildupAgent = True
        with open(st.session_state['agent_saved_path'], 'r', encoding='utf-8') as file:
            data = json.load(file)
            for agent_config in data["agent_configs"]:
                st.session_state.list_items.append(agent_config["name"])

    # Display the dynamic list
    st.write(":woman-woman-boy-boy: 参与的代理: ")
    for item in st.session_state.list_items:
        st.write("- ", item)

    # Section 2 用户研究
    st.subheader(" :man-woman-boy-boy: 用户研究模块")
    user1des = st.text_area(" :male-office-worker: 输入用户1描述", height=100)
    user2des = st.text_area(" :female-office-worker: 输入用户2描述", height=100)
    user3des = st.text_area(" :woman-girl: 输入用户3描述", height=100)
    if st.button("用户Agent生成",use_container_width=True):
        usernum = 1
        if len(user1des) > 1 :
            username = buildUpUser(user1des, usernum)
            st.session_state.list_items.append("User1")
            usernum += 1
        if len(user2des) > 1:
            username = buildUpUser(user2des, usernum)
            st.session_state.list_items.append("User2")
            usernum += 1
        if len(user3des) > 1:
            username = buildUpUser(user3des, usernum)
            st.session_state.list_items.append("User3")
            usernum += 1
        st.write("生成用户代理成功")

    if st.button("发起方案询问", use_container_width=True):
        chatWithUser(task_description)

    # Section 3 设计咨询模块
    st.subheader(" :derelict_house_building: 设计咨询模块")

    checkrecord = []
    for item in st.session_state.list_items:
        bnum = st.checkbox(item)
        checkrecord.append(bnum)

    index = -1
    for i in range(len(checkrecord)):
        if checkrecord[i]:
            index = i
            break

    question = st.text_area("输入咨询内容", height=100)

    st.markdown("---")
    if st.button("单个代理咨询", use_container_width=True):
        if index < 0:
            st.write("请选择有效代理")
        else:
            chatwithsingleUser(index, question)

    st.markdown("---")
    if st.button("群体咨询", use_container_width=True):
        chatwithMultipleuser(task_description, question)

    # with CaptureStdout() as capture:
    #     print("Hello, console!")  # This will be captured and not printed to the terminal
    #     # Add more console print statements or function calls that print to console here
    #
    #     # Display the captured console output in the Streamlit app
    # st.code(capture.value, language='text')



if __name__ == "__main__":
    main()