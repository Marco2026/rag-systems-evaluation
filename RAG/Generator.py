import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re

class Generator():
    
    def __init__(self, model_name: str, system_prompt: str, device: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.system_prompt = system_prompt
        self.device = device
        self.prepare_model()


    def prepare_model(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        generator_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
            add_eos_token=True
        )

        self.model = generator_model
        self.tokenizer = tokenizer

    
    def ask_generator(self, user_prompt: str):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=300, 
            do_sample=False
        )
        answer = self.clean_output(inputs, outputs)
        return answer

    def clean_output(self, inputs, outputs):
        input_length = inputs.input_ids.shape[-1]
        generated_tokens = outputs[0][input_length:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        answer = re.sub(r'^[a-zA-Z]+\n', '', answer).lstrip()
        answer = answer.strip()
        for prefix in ("user\n", "system\n", "assistant\n", "user ", "system ", "assistant "):
            answer = answer.removeprefix(prefix)
        return answer.strip()