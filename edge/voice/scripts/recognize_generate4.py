"""
this file is intended to recieve the audio from the user and then generate the response according to the audio
there are to options for generating the answer, one is using the self_deployed llama model,
the other is using the api proveide by ali 通义千问
note that if the answer generated is not likely to be correct or even not generated,
the program will inform the user its incapability

in consideration of time, we intend to response specific answer with pre_prepared audio,
and the general answer with llama or 通义千问
"""

import json
import logging
import os
import random
import socket
import threading
import time
from http import HTTPStatus

import dashscope
import requests
import speech_recognition as sr
from dashscope import Generation
from huaweicloud_sis.bean.asr_request import SasrWebsocketRequest
from huaweicloud_sis.bean.callback import RasrCallBack
from huaweicloud_sis.bean.sis_config import SisConfig
from huaweicloud_sis.bean.tts_request import TtsCustomRequest
from huaweicloud_sis.client.asr_client import SasrWebsocketClient
from huaweicloud_sis.client.tts_client import TtsCustomizationClient
from pydub import AudioSegment
from pydub.playback import play

logger = logging.getLogger(__name__)

record_socket = None


def establish_record_connection():
    global record_socket
    while True:
        try:
            record_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            record_socket.connect(("localhost", 5333))
            logger.info("Connected to localhost:5333")
            break
        except ConnectionRefusedError:
            logger.warning("Connection refused. Retrying in 1 second.")
            time.sleep(1)


class SARSCallBack(RasrCallBack):
    """回调类，用户需要在对应方法中实现自己的逻辑，其中on_response必须重写"""

    def __init__(self):
        self.response_data = None  # 用于存储享用数据的属性

    def on_open(self):
        """websocket连接成功会回调此函数"""
        logger.info("websocket connect success")

    def on_start(self, message):
        """
            websocket 开始识别回调此函数
        :param message: 传入信息
        :return: -
        """
        logger.info("websocket start to recognize, %s", message)

    def on_response(self, message):
        """
            websockert返回响应结果会回调此函数
        :param message: json格式
        :return: -
        """
        logger.debug(json.dumps(message, indent=2, ensure_ascii=False))
        self.response_data = json.dumps(message, indent=2, ensure_ascii=False)

    def on_end(self, message):
        """
            websocket 结束识别回调此函数
        :param message: 传入信息
        :return: -
        """
        logger.info("websocket is ended, %s", message)

    def on_close(self):
        """websocket关闭会回调此函数"""
        logger.info("websocket is closed")

    def on_error(self, error):
        """
            websocket出错回调此函数
        :param error: 错误信息
        :return: -
        """
        logger.error("websocket meets error, the error is %s", error)

    def on_event(self, event):
        """
            出现事件的回调
        :param event: 事件名称
        :return: -
        """
        logger.debug("receive event %s", event)

    def get_response_data(self):
        return self.response_data


def close_websocket(client):
    try:
        client.close()
    except Exception as e:
        logger.error("Error closing websocket: %s", e)


def generate_random_uint64():
    # 生成一个无符号64位整数
    random_uint64 = random.getrandbits(64)
    return int(random_uint64)


def qianwen(input_question: str):
    """add another option for"""
    dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY")
    resp = Generation.call(
        model="qwen-v1",
        prompt=input_question,
        top_k=88.0,
        top_p=0.8,
        seed=int(generate_random_uint64() % 9223372036854775807),
    )
    # The response status_code is HTTPStatus.OK indicate success,
    # otherwise indicate request is failed, you can get error code
    # and message from code and message.
    if resp.status_code == HTTPStatus.OK:
        logger.info("Qianwen output: %s", resp.output)
        logger.debug("Qianwen usage: %s", resp.usage)
        return resp.output.get("text")
    else:
        logger.error("Qianwen error code: %s", resp.code)
        logger.error("Qianwen error message: %s", resp.message)
        return None


class VoiceInteraction:
    """
    useage: before the whole program is initiated,
     you need first initiated the llama.cpp/server so that the server will be initiated,
     then,open a terminal and type in the following command:
    `ssh -f -N -L 8080:localhost:8080 zhoujian@222.20.97.89 -p 11224` and then type in the password of the server
    if the port 8080 is not available on server "zhoujian",then pick another port
    and change the first 8080 to the port you choose,
    if the port 8080 is not available on local device, choose another port and change the second 8080 in the command
    either change should be made in the `__init__` method below as well
    """

    def __init__(
        self,
        ak=None,  # 用户的ak, 通过环境变量 HUAWEICLOUD_AK 配置
        sk=None,  # 用户的sk, 通过环境变量 HUAWEICLOUD_SK 配置
        region="cn-north-4",  # region，如cn-north-4
        project_id=None,
        # 同region一一对应，参考https://support.huaweicloud.com/api-sis/sis_03_0008.html
        local_port=19327,
        server_port=19327,
    ):
        self.ak = ak or os.environ.get("HUAWEICLOUD_AK")
        self.sk = sk or os.environ.get("HUAWEICLOUD_SK")
        self.region = region
        self.project_id = project_id or os.environ.get("HUAWEICLOUD_PROJECT_ID")

        self.local_port = local_port
        self.server_port = server_port
        self.chat = self.llama

    def close_websocket(self, client):
        try:
            client.close()
        except Exception as e:
            logger.error("Error closing websocket: %s", e)

    def SASR(self, audio, audio_format="pcm16k16bit", property="chinese_16k_general"):
        """
        语音识别函数，
        :param audio，传入一个wav文件的路径，或者是直接的音频流
        :return:str,the text of the inference result
        """
        # step1 初始化SasrWebsocketClient
        my_callback = SARSCallBack()
        config = SisConfig()
        # 设置连接超时,默认是10
        config.set_connect_timeout(10)
        # 设置读取超时, 默认是10
        config.set_read_timeout(10)
        # 设置connect lost超时，一般在普通并发下，不需要设置此值。默认是10
        config.set_connect_lost_timeout(10)
        # websocket暂时不支持使用代理
        sasr_websocket_client = SasrWebsocketClient(
            ak=self.ak,
            sk=self.sk,
            use_aksk=True,
            region=self.region,
            project_id=self.project_id,
            callback=my_callback,
            config=config,
        )
        try:
            # step2 构造请求
            request = SasrWebsocketRequest(audio_format, property)
            # 所有参数均可不设置，使用默认值
            request.set_add_punc("yes")  # 设置是否添加标点， yes or no， 默认no
            request.set_interim_results("no")  # 设置是否返回中间结果，yes or no，默认no
            request.set_digit_norm("no")  # 设置是否将语音中数字转写为阿拉伯数字，yes or no，默认yes
            # request.set_vocabulary_id('')     # 设置热词表id，若不存在则不填写，否则会报错
            request.set_need_word_info("no")  # 设置是否需要word_info，yes or no, 默认no

            # step3 连接服务端
            sasr_websocket_client.sasr_stream_connect(request)

            # step4 发送音频
            sasr_websocket_client.send_start()
            sasr_websocket_client.send_audio(audio)
            # 连续模式下，可多次发送音频，发送格式为byte数组
            if isinstance(audio, str):  # if this is a string ,then we need to open and read it
                with open(audio, "rb") as f:
                    data = f.read()
                    sasr_websocket_client.send_audio(data)  # 可选byte_len和sleep_time参数，建议使用默认值
            # if it is not a string ,we just need to send it
            sasr_websocket_client.send_end()
        except Exception as e:
            logger.error("sasr websocket error: %s", e)
        finally:
            # step5 关闭客户端，使用完毕后一定要关闭，否则服务端20s内没收到数据会报错并主动断开。
            close_thread = threading.Thread(target=self.close_websocket, args=(sasr_websocket_client,))
            close_thread.start()
            response_data = my_callback.get_response_data()
            if response_data is None:
                return None
            else:
                data = json.loads(response_data)
                text = ""
                for i in range(len(data["segments"])):
                    text += data["segments"][i]["result"]["text"]
                    if data["segments"][i]["result"]["score"] < 0.5:
                        return None  # 如果识别的结果可信度太低的话干脆就等于识别失败算了
                return text

    def llama(self, input_question: str, max_tokens=45, temperature=0.7, num_beams=4, top_k=40):
        """
        use llama model to generate response according to the given text
        :param input_question: str, the text that is going to be asked to the llama model
        :return: str,the text that is generated by the llama model
        """
        url = f"http://localhost:{self.local_port}/v1/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "prompt": input_question,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "num_beams": num_beams,
            "top_k": top_k,
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            choices = result.get("choices", [])
            text_list = [choice.get("text", "") for choice in choices]
            result_text = " ".join(text_list)
            return result_text
        else:
            logger.error("request failed with status code: %s", response.status_code)
            return None

    def TTSC(
        self,
        given_text,
        temp_path="./temp.wav",
        property="chinese_huaxiaoru_common",
        audio_format="wav",
        sample_rate="8000",
        volume=50,
        pitch=0,
        speed=0,
    ):
        """
        speech synthesis function that generate and play an audio from a given text
        :param given_text: the text that is going to be generated into an audio and played
        :param temp_path: the path that is used to temporarily store the wav and will be deleted
            when tht audio is played
        """

        text = given_text
        config = SisConfig()
        config.set_connect_timeout(10)  # 设置连接超时，单位s
        config.set_read_timeout(10)  # 设置读取超时，单位s
        # 设置代理，使用代理前一定要确保代理可用。 代理格式可为[host, port] 或 [host, port, username, password]
        # config.set_proxy(proxy)
        ttsc_client = TtsCustomizationClient(self.ak, self.sk, self.region, self.project_id, sis_config=config)
        # step2 构造请求
        ttsc_request = TtsCustomRequest(text)
        # 设置请求，所有参数均可不设置，使用默认参数
        # 设置属性字符串， language_speaker_domain, 默认chinese_xiaoyan_common, 参考api文档
        ttsc_request.set_property(property)
        # 设置音频格式，默认wav，可选mp3和pcm
        ttsc_request.set_audio_format(audio_format)
        # 设置采样率，8000 or 16000, 默认8000
        ttsc_request.set_sample_rate(sample_rate)
        # 设置音量，[0, 100]，默认50
        ttsc_request.set_volume(volume)
        # 设置音高, [-500, 500], 默认0
        ttsc_request.set_pitch(pitch)
        # 设置音速, [-500, 500], 默认0
        ttsc_request.set_speed(speed)
        # 设置是否保存，默认False
        ttsc_request.set_saved(True)
        # 设置保存路径，只有设置保存，此参数才生效
        ttsc_request.set_saved_path(temp_path)
        # step3 发送请求，返回结果。如果设置保存，可在指定路径里查看保存的音频。
        ttsc_client.get_ttsc_response(ttsc_request)
        # 使用AudioSegment不正确无法在板子上跑
        song = AudioSegment.from_wav(temp_path)
        play(song)
        # subprocess.call(["aplay",temp_path])
        os.remove(temp_path)

    def alert(self, status):
        """
        when passed in a status, generate a speech to alert the driver
        :param status: int , the status of the driver
        """
        if status == 1:
            response_text = self.chat("在50个字以内告诉我开车的时候不能闭眼睛超过3秒以上的重要性")
            index = response_text.find("。")
            if index != -1:
                response_text = response_text[: index + 1]
            flag = False
            for i in range(2):
                if len(response_text) >= 5:
                    flag = True
                    break
                song = AudioSegment.from_wav(
                    "/home/jetson/Documents/VoiceInteraction/audio_bags/让我仔细想一想怎样劝诫你是最有效的.wav"
                )
                play(song)
                response_text = self.chat("在50个字以内告诉我开车的时候不能闭眼睛超过3秒以上的重要性")
            if flag:
                logger.info("response_text:%s", response_text)
                self.TTSC(response_text)
            else:
                song = AudioSegment.from_wav(
                    "/home/jetson/Documents/VoiceInteraction/audio_bags/开车的时候不可以一直闭眼哦.wav"
                )
                play(song)
        elif status == 2:
            response_text = self.chat("在50个字以内告诉我开车的时候不能打哈欠超过3秒以上的重要性")
            index = response_text.find("。")
            if index != -1:
                response_text = response_text[: index + 1]
            flag = False
            for i in range(2):
                if len(response_text) >= 5:
                    flag = True
                    break
                song = AudioSegment.from_wav(
                    "/home/jetson/Documents/VoiceInteraction/audio_bags/让我仔细想一想怎样劝诫你是最有效的.wav"
                )
                play(song)
                response_text = self.chat("在50个字以内告诉我开车的时候不能打哈欠超过3秒以上的重要性")
            if flag:
                logger.info("response_text:%s", response_text)
                self.TTSC(response_text)
            else:
                # todo 播放音频： 开车的时候如果你一直打哈欠会很危险哦
                song = AudioSegment.from_wav(
                    "/home/jetson/Documents/VoiceInteraction/audio_bags/开车的时候如果你一直打哈欠会很危险哦.wav"
                )
                play(song)
        elif status == 3:
            response_text = self.chat("在50个字以内告诉我开车的时候不能打电话的重要性")
            index = response_text.find("。")
            if index != -1:
                response_text = response_text[: index + 1]
            flag = False
            for i in range(2):
                if len(response_text) >= 5:
                    flag = True
                    break
                song = AudioSegment.from_wav(
                    "/home/jetson/Documents/VoiceInteraction/audio_bags/让我仔细想一想怎样劝诫你是最有效的.wav"
                )
                play(song)
                response_text = self.chat("在50个字以内告诉我开车的时候不能打电话的重要性")
            if flag:
                logger.info("response_text:%s", response_text)
                self.TTSC(response_text)
            else:
                song = AudioSegment.from_wav(
                    "/home/jetson/Documents/VoiceInteraction/audio_bags/千万不要边打电话边开车哦.wav"
                )
                play(song)
        elif status == 4:
            response_text = self.chat("在50个字以内告诉我开车的时候不要左顾右盼的重要性")
            index = response_text.find("。")
            if index != -1:
                response_text = response_text[: index + 1]
            flag = False
            for i in range(2):
                if len(response_text) >= 5:
                    flag = True
                    break
                song = AudioSegment.from_wav(
                    "/home/jetson/Documents/VoiceInteraction/audio_bags/让我仔细想一想怎样劝诫你是最有效的.wav"
                )
                play(song)
                response_text = self.chat("在50个字以内告诉我开车的时候不能频繁左顾右盼的重要性")
            if flag:
                logger.info("response_text:%s", response_text)
                self.TTSC(response_text)
            else:
                song = AudioSegment.from_wav(
                    "/home/jetson/Documents/VoiceInteraction/audio_bags/频繁的左顾右盼可不是一个开车的好习惯.wav"
                )
                play(song)

    def asked(self, audio) -> int:
        """
        when passed in the path of audio, analyze it and generate respond to it.
        :param audio: the path of the audio
        """
        global record_socket
        inference_text = self.SASR(audio)
        if inference_text is None:
            song = AudioSegment.from_wav("/home/jetson/Documents/VoiceInteraction/audio_bags/再说一次.wav")
            play(song)
            return 1
        if "记录" in inference_text:
            # todo smartvideorecord
            if record_socket is not None:
                try:
                    record_socket.sendall("record".encode())
                except socket.error:
                    logger.error("Failed to send record command")
                # record_socket.shutdown(socket.SHUT_WR)
            else:
                logger.warning("record socket is none")
            song = AudioSegment.from_wav("/home/jetson/Documents/VoiceInteraction/audio_bags/开始录制.wav")
            play(song)
            return 1
        if "再见" in inference_text:
            song = AudioSegment.from_wav("/home/jetson/Documents/VoiceInteraction/audio_bags/再见.wav")
            play(song)
            return 0
        if "语音助手" in inference_text:
            text = "在25字内回答我的问题："
            inference_text = inference_text.replace("语音助手", "")
            text += inference_text
            logger.debug("Prompt text: %s", text)
            response = self.chat(text)
            flag = False
            logger.debug("Response length=%d, content=%s", len(response), response)
            for i in range(5):
                if len(response) >= 5:
                    flag = True
                    break
                song = AudioSegment.from_wav(
                    "/home/jetson/Documents/VoiceInteraction/audio_bags/请让我仔细思考一下.wav"
                )
                play(song)
                response = self.chat(text)
            if not flag:
                song = AudioSegment.from_wav("/home/jetson/Documents/VoiceInteraction/audio_bags/想不出来.wav")
                play(song)
                pass
            else:
                self.TTSC(response, "./temp.wav")
            return 1

    def communicate(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            logger.info("请说话")
            recognizer.adjust_for_ambient_noise(source)
            while True:
                try:
                    audio = recognizer.listen(source, timeout=5)
                    logger.debug("finish recording, saving")
                    audio_pcm16k16bit = audio.get_raw_data(convert_width=2, convert_rate=16000)
                    result = self.asked(audio_pcm16k16bit)
                    if result == 0:
                        return
                except sr.WaitTimeoutError:
                    logger.warning("录音超时，未检测到声音")
                except sr.RequestError as e:
                    logger.error("请求错误：%s", e)
                except sr.UnknownValueError:
                    logger.warning("无法识别语音")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    vi = VoiceInteraction()
    vi.communicate()
