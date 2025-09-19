#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenAI-Compatible API Client

SETUP INSTRUCTIONS:
1. Replace "YOUR_API_KEY_HERE" with your actual API key
2. Replace "YOUR_API_BASE_URL" with your API base URL (e.g., "https://api.openai.com/v1")
3. Configure model and other parameters as needed

USAGE EXAMPLES:

# Basic usage with OpenAI API
client = OpenAIClient(
    app_key="sk-your-openai-api-key",
    base_url="https://api.openai.com/v1"
)

# Single request
response = client.server_by_openai({"prompt": "Hello, world!"})

# Batch processing with multiple threads
requests = [{"prompt": f"Question {i}"} for i in range(10)]
responses = client.n_threads_do(num_threads=3, data_list=requests)
"""

import requests
import json
import time
import threading
import queue
import random
from typing import Dict, Any, List, Optional, Tuple

class OpenAIClient:
    """OpenAI-compatible API client with multi-threading support"""
    
    def __init__(self, app_key="YOUR_API_KEY_HERE", base_url="YOUR_API_BASE_URL", 
                 max_tokens=16384, timeout=60*1000*5):
        """
        Initialize the API client
        
        Args:
            app_key: API key for authentication
            base_url: Base URL for the API service
            max_tokens: Maximum tokens in response
            timeout: Request timeout in milliseconds
        """
        if app_key == "YOUR_API_KEY_HERE":
            raise ValueError("Please replace 'YOUR_API_KEY_HERE' with your actual API key")
        if base_url == "YOUR_API_BASE_URL":
            raise ValueError("Please replace 'YOUR_API_BASE_URL' with your actual API base URL")
            
        self.appId = app_key
        self.base_url = base_url.rstrip('/')
        self.max_tokens = max_tokens
        self.timeout = timeout
    
    def server_by_openai(self, raw_request: Dict[str, Any], msg_id=None):
        """Send a single request to the API"""
        prompt = raw_request["prompt"]
        msg = [{"role": "user", "content": prompt}]
        
        try:
            answer = self._do_post_openai(msg, msg_id)
            if answer is None:
                answer = "返回异常，无结果。"
        except Exception as e:
            print(f"异常信息: {e}")
            answer = "执行异常。"
            
        return answer

    def _do_post_openai(self, msg, msg_id=None):
        """Make API request with retry logic"""
        retry_times = 0
        response = None

        max_retry = 100
        # Retry until success or max attempts reached
        while retry_times < max_retry:
            response = self._call_api(msg, msg_id)
            if response is not None:
                break
            retry_times = retry_times + 1
            print('Query失败，进行第{}次重试'.format(retry_times))
            time.sleep(3) 
        return response

    def _call_api(self, msg, msg_id=None):
        """Make the actual API call"""
        url = f'{self.base_url}/chat/completions'
        headers = {
            'Authorization': f'Bearer {self.appId}',
            'Content-Type': 'application/json',
        }
        
        data = {
            "messages": msg,
            "model": "gpt-4o-2024-11-20",  # Default model, can be configured
            "stream": False,
            "max_tokens": self.max_tokens,
        }
        
        try:
            ret = requests.post(url=url, data=json.dumps(data), headers=headers, timeout=int(self.timeout/1000))

            status_dict = {
                200: 'success', 
                400: 'Bad Request', 
                401: 'Unauthorized',
                408: 'Request Timeout', 
                429: 'Too Many Requests', 
                450: 'Content Policy Violation', 
                451: 'Content Policy Violation', 
                500: 'Internal Server Error', 
                504: 'Timeout'
            }
            
            if msg_id:
                status_msg = status_dict.get(ret.status_code, f'Unknown status: {ret.status_code}')
                print(msg_id, ret.status_code, status_msg)
            
            if ret.status_code == 200:
                result = json.loads(ret.content)
                return result["choices"][0]["message"]["content"]
            elif ret.status_code in [400]:
                return "fail: over context window"
            elif ret.status_code in [450, 451]:
                return f'fail: Content policy violation, status:{ret.status_code}'
            return None
            
        except Exception as e:
            if msg_id:
                print(f"{msg_id}: Request exception: {e}")
            return None

    def n_threads_do(self, num_threads, data_list):
        """
        使用多线程请求API

        Args:
            num_threads (int): 线程数
            data_list (List[dict]): 已经构造好的Prompt数据，放在list里面
        Returns:
            List[str]: API的处理结果
        """
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
            # 处理数据
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


def main():
    """Example usage"""
    print("OpenAI-Compatible API Client")
    print("=" * 50)
    
    try:
        # Example configuration - users need to replace these values
        client = OpenAIClient(
            app_key="YOUR_API_KEY_HERE",  # Replace with actual API key
            base_url="YOUR_API_BASE_URL"   # Replace with actual base URL
        )
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("\nPlease configure the client with your actual API credentials:")
        print("1. Set your API key (e.g., 'sk-your-openai-api-key')")
        print("2. Set your base URL (e.g., 'https://api.openai.com/v1')")
        return
    
    # Example single request
    print("\nExample usage:")
    print("client = OpenAIClient(")
    print("    app_key='your-api-key',")
    print("    base_url='https://api.openai.com/v1'")
    print(")")
    print("response = client.server_by_openai({'prompt': 'Hello!'})")


if __name__ == "__main__":
    main()
