import os
import openai
import pickle
from collections import defaultdict
from utils.io.io_utils import add_to_log

openai.api_key = 'YOUR_KEY_HERE'


def create_or_load_cache(cache_file):
    if not os.path.exists(os.path.dirname(cache_file)):
        # add_to_log(f"Creating directory for {cache_file}")
        os.makedirs(os.path.dirname(cache_file))

    cache: defaultdict[str, dict] = defaultdict(dict)
    if os.path.exists(cache_file):
        # add_to_log(f"loading cache from {cache_file}")
        cache = pickle.load(open(cache_file, "rb"))
    return cache


def query_LLM(prompt, stop_sequences, cache_file):
    cache = create_or_load_cache(cache_file)
    response = cache[prompt].get("gpt-4", None)
    if response is None:
        success = False
        max_retry = 3
        while not success and max_retry > 0:
            try:
                response = openai.ChatCompletion.create(
                    model='gpt-4',
                    messages=[{'role':'system', 'content':prompt}],
                    temperature=1,
                    stop=stop_sequences,
                    max_tokens=3000
                )
                success = True
            except Exception as e:
                print("Error encountered")
                max_retry -= 1
                if max_retry == 0:
                    raise e
                else:
                    print(e)
        cache[prompt]["gpt-4"] = response
    pickle.dump(cache, open(cache_file, "wb"))
    response.text = response.choices[0].message['content']
    return response



if __name__ == "__main__":
    prompt = 'say hi'
    stop_sequences = []
    cache_file = 'cache/llm_test_correction.pkl'
    response = query_LLM(prompt, stop_sequences, cache_file)
    print(response['usage'])
    print(response.choices[0].message['content'])
    print(response.text)