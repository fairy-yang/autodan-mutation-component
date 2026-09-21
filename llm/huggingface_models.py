from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json


class HuggingFaceModel:
    def __init__(self, repo_name: str, config_dir: str, config_name: str, token=None):
        """
        Initialize the Hugging Face model class in a distributed manner.

        Args:
            repo_name (str): Name of the Hugging Face model repository, e.g., "meta-llama/Meta-Llama-3-8B".
            config_dir (str): Directory where your config and template files are located.
            config_name (str): Name of the config file.
            token (str): Hugging Face API token for private models.
        """
        print(f"Checking for model in '{config_dir}/model_ckpt'...")
        model_dir = f"{config_dir}/model_ckpt"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, repo_name.replace("/", "_"))
        if not os.path.exists(model_path):
            print(f"Model not found in {model_path}. Downloading from Hugging Face...")
            AutoModelForCausalLM.from_pretrained(repo_name, token=token).save_pretrained(model_path)
            AutoTokenizer.from_pretrained(repo_name, token=token).save_pretrained(model_path)
            print(f"Model downloaded and saved to {model_path}.")
        else:
            print(f"Model found in {model_path}. Using cached model.")
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto"
        )
        self.config = json.load(open(f'{config_dir}/generation_configs/{config_name}.json'))
        chat_template = open(f'{config_dir}/{self.config["chat_template"]}').read()
        chat_template = chat_template.replace('    ', '').replace('\n', '')
        self.tokenizer.chat_template = chat_template
        self.last_generation_stats = {}
        print("Model loaded with automatic device mapping across GPUs.")

    def _context_window(self):
        for value in (
            getattr(self.model.config, "max_position_embeddings", None),
            getattr(self.tokenizer, "model_max_length", None),
        ):
            if isinstance(value, int) and 0 < value < 10**7:
                return value
        return None

    @staticmethod
    def _truncate_token_ids(input_ids, budget: int, strategy: str):
        if input_ids.shape[-1] <= budget:
            return input_ids

        if strategy == "head":
            return input_ids[..., :budget]
        if strategy == "tail":
            return input_ids[..., -budget:]

        head = budget // 2
        tail = budget - head
        return torch.cat([input_ids[..., :head], input_ids[..., -tail:]], dim=-1)

    def _prepare_inputs(
        self,
        plain_text: str,
        max_new_tokens: int,
        max_context_tokens: int = None,
        allow_input_truncation: bool = True,
        truncation_strategy: str = "middle",
    ):
        inputs = self.tokenizer(plain_text, return_tensors="pt")
        input_tokens = int(inputs["input_ids"].shape[-1])

        context_window = max_context_tokens or self._context_window()
        prompt_budget = None
        truncated = False

        if context_window:
            prompt_budget = max(int(context_window) - int(max_new_tokens), 1)
            if input_tokens > prompt_budget:
                if not allow_input_truncation:
                    raise ValueError(
                        "Input prompt is too long for the configured context window: "
                        f"{input_tokens} prompt tokens > {prompt_budget} budget tokens "
                        f"(context={context_window}, max_new_tokens={max_new_tokens})."
                    )
                inputs["input_ids"] = self._truncate_token_ids(
                    inputs["input_ids"],
                    prompt_budget,
                    truncation_strategy,
                )
                inputs["attention_mask"] = self._truncate_token_ids(
                    inputs["attention_mask"],
                    prompt_budget,
                    truncation_strategy,
                )
                truncated = True

        self.last_generation_stats = {
            "input_tokens": input_tokens,
            "used_input_tokens": int(inputs["input_ids"].shape[-1]),
            "max_new_tokens": int(max_new_tokens),
            "context_window": context_window,
            "prompt_budget": prompt_budget,
            "truncated": truncated,
            "truncation_strategy": truncation_strategy if truncated else None,
        }

        return {k: v.to(self.model.device) for k, v in inputs.items()}

    def generate(self, system: str, user: str, max_length: int = 1000, **kwargs):
        """
        Generate a response based on the input text.
        """
        messages = [
            {'role': 'system', 'content': f'{system}'},
            {'role': 'user', 'content': f'{user}'},
        ]
        plain_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        max_new_tokens = kwargs.pop("max_new_tokens", None)
        if max_new_tokens is None:
            max_new_tokens = max(max_length - 1, 1)

        inputs = self._prepare_inputs(
            plain_text,
            max_new_tokens=max_new_tokens,
            max_context_tokens=kwargs.pop("max_context_tokens", None),
            allow_input_truncation=kwargs.pop("allow_input_truncation", True),
            truncation_strategy=kwargs.pop("truncation_strategy", "middle"),
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )
        response_start = inputs["input_ids"].shape[-1]
        response_ids = outputs[0][response_start:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return response

    def continue_generate(self, system: str, user1: str, assistant1: str, user2: str, max_length: int = 1000, **kwargs):
        """
        Continue a conversation and generate a response.
        """
        messages = [
            {'role': 'system', 'content': f'{system}'},
            {'role': 'user', 'content': f'{user1}'},
            {'role': 'assistant', 'content': f'{assistant1}'},
            {'role': 'user', 'content': f'{user2}'},
        ]
        plain_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        max_new_tokens = kwargs.pop("max_new_tokens", None)
        if max_new_tokens is None:
            max_new_tokens = max(max_length - 1, 1)

        inputs = self._prepare_inputs(
            plain_text,
            max_new_tokens=max_new_tokens,
            max_context_tokens=kwargs.pop("max_context_tokens", None),
            allow_input_truncation=kwargs.pop("allow_input_truncation", True),
            truncation_strategy=kwargs.pop("truncation_strategy", "middle"),
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )
        response_start = inputs["input_ids"].shape[-1]
        response_ids = outputs[0][response_start:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return response

    def conditional_generate(self, condition: str, system: str, user: str, max_length: int = 1000, **kwargs):
        """
        Generate a response with additional conditions appended to the input prompt.
        """
        messages = [
            {'role': 'system', 'content': f'{system}'},
            {'role': 'user', 'content': f'{user}'},
        ]
        plain_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        plain_text += condition

        max_new_tokens = kwargs.pop("max_new_tokens", 512)
        inputs = self._prepare_inputs(
            plain_text,
            max_new_tokens=max_new_tokens,
            max_context_tokens=kwargs.pop("max_context_tokens", None),
            allow_input_truncation=kwargs.pop("allow_input_truncation", True),
            truncation_strategy=kwargs.pop("truncation_strategy", "middle"),
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )
        response_start = inputs["input_ids"].shape[-1]
        response_ids = outputs[0][response_start:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return response
