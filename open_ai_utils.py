import openai
from typing import Dict, Any
import time
import requests
import json

import threading
import queue
import random
from typing import List, Tuple

# 请求OpenAI的API
class OpenAIClient:
    # appkey是每个组一个，有总的RPM限制
    def __init__(self, app_key=1791012913151668297, max_tokens=16384, timeout=60*1000*5):
        self.appId = app_key
        self.max_tokens = max_tokens
        self.timeout = timeout
    
    # 单次请求g4o
    def server_by_openai(self, raw_request: Dict[str, Any], msg_id=None):
        prompt = raw_request["prompt"]
        msg = [{"role": "user", "content": prompt}]
        
        # 发送POST请求
        try:
            answer = self._do_post_openai(msg, msg_id)
            if answer is None:
                answer = "返回异常，无结果。"
        except Exception as e:
            print(f"异常信息: {e}")
            answer = "执行异常。"
            
        return answer

    def _do_post_openai(self, msg, msg_id=None):
        retry_times = 0
        response = None

        max_retry = 100
        # 没有最大次数限制，retry直到success
        #for i in range(0, max_retry):
        while True:
            response = self._call_internal_api(msg, msg_id)
            if response is not None:
                break
            retry_times = retry_times + 1
            print('Query失败，进行第{}次重试'.format(retry_times))
            time.sleep(3) 
        return response

    def _call_internal_api(self, msg, msg_id=None):
        url = 'https://aigc.sankuai.com/v1/openai/native/chat/completions'
        headers = {
            'Authorization': f'Bearer {self.appId}',
            'Content-Type': 'application/json',
        }
        # 在这里可以设置一些超参数
        data = {
            "messages": msg,
            #"model": "gpt-4o-eva", 
            'model': 'gpt-4o-2024-11-20',
            "stream": False,
            "max_tokens": 16384,
        }
        # print(data['model'])
        ret = requests.post(url=url, data=json.dumps(data), headers=headers, timeout=int(self.timeout))

        status_dict = {200:'success', 400:'Bad Request', 408:'Request Timeout', 429:'Too Many Requests', 
            450:'input没有通过保时洁', 451:'output没有通过保时洁', 500:'Internal Server Error', 504:'Timeout'}
        print(msg_id, ret.status_code, status_dict[ret.status_code])
        
        if ret.status_code == 200:
            result = json.loads(ret.content)
            return result["choices"][0]["message"]["content"]
        elif ret.status_code in [400]:
            return "fail: over context window"
        elif ret.status_code in [450, 451]:
            return f'fail: 没有通过保时洁，status:{ret.status_code}'
        return None

    '''--------------[START]封装一个多线程请求g4o的接口[START]--------------'''
    def n_threads_do(self, num_threads, data_list):
        '''
        使用多线程请求g4o

        Args:
            num_threads (int): 线程数
            data_list (List[dict]): 已经构造好的Prompt数据，放在list里面
        Returns:
            List[str]: g4o的处理结果
        '''
        # 创建一个队列用于存储结果，线程安全队列
        result_queue = queue.Queue()
        # 将数据按照线程数切分
        data_splits = self._split_data(data_list, num_threads)
        threads = []
        
        print(f"Starting processing with {num_threads} threads")
        print(f"Total items: {len(data_list)}")
        print(f"Items per thread: ~{len(data_list) // num_threads}")
        
        # 创建并启动线程
        for i, data_batch in enumerate(data_splits):
            thread_name = f"Thread-{i+1}"
            thread = threading.Thread(
                target=self._process_data_batch,
                args=(data_batch, result_queue, thread_name)
            )
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 从队列中收集所有结果
        all_results = []
        while not result_queue.empty():
            thread_result = result_queue.get()
            all_results.extend(thread_result['batch_results'])
        
        # 按原始顺序排序结果
        all_results.sort(key=lambda x: x['index'])
        print('Processing all batches done')
        return [item['output'] for item in all_results]
    
    def _process_data_batch(self, indexed_batch, result_queue, thread_name):
        """
        处理一批数据的函数
        :param indexed_batch: 要处理的一批数据，每个元素是(索引, 数据)的元组
        :param thread_name: 线程名称
        """
        
        batch_results = []
        # 处理批次中的每个数据
        for idx, data in indexed_batch:
            # 处理数据（这里简单地在字符串后面加上处理标记）
            result = self.server_by_openai(data)
            batch_results.append({
                'index': idx,
                'input': data,
                'output': result
            })
        
        # 将整个批次的结果放入队列
        result_queue.put({
            'thread_name': thread_name,
            'batch_results': batch_results
        })
        
    
    def _split_data(self, data_list, num_threads):
        """
        将数据列表均匀分配给指定数量的线程，并为每个数据项添加索引信息，返回诸如List[List[Tuple[int, str]]]形式的数据
        """
        # 首先为数据添加索引
        indexed_data = list(enumerate(data_list))
        
        batch_size = len(data_list) // num_threads
        remainder = len(data_list) % num_threads
        
        result = []
        start = 0
        for i in range(num_threads):
            current_batch_size = batch_size + (1 if i < remainder else 0)
            end = start + current_batch_size
            result.append(indexed_data[start:end])
            start = end
        
        return result
    '''--------------[END]封装一个多线程请求g4o的接口[END]--------------'''
