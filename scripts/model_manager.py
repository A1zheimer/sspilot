import gc
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelManager:
    def __init__(self, config: dict):
        self.config = config
        self.models = {}
        self.tokenizers = {}
        mem_cfg = config.get("MEMORY_SAFETY", {})
        self.min_free_before_load = mem_cfg.get("min_free_before_load", 10.0)
        self.unload_wait_max = mem_cfg.get("unload_wait_max", 30)
        self.unload_wait_interval = mem_cfg.get("unload_wait_interval", 2)

    def get_free_memory_gb(self) -> float:
        """DGX Spark UVM: torch.cuda.mem_get_info returns 0, use psutil instead."""
        try:
            import psutil
            return psutil.virtual_memory().available / (1024 ** 3)
        except ImportError:
            pass
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            if free > 0:
                return free / (1024 ** 3)
        return 0.0

    def get_used_memory_gb(self) -> float:
        """DGX Spark UVM: use psutil for accurate memory tracking."""
        try:
            import psutil
            vm = psutil.virtual_memory()
            return vm.used / (1024 ** 3)
        except ImportError:
            pass
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            if total > 0:
                return (total - free) / (1024 ** 3)
        return 0.0

    def safe_unload(self, name: str, required_free_gb: float = 0) -> bool:
        print(f"[ModelManager] 开始卸载 {name}...")
        if name in self.models:
            try:
                self.models[name].cpu()
            except Exception:
                pass
            del self.models[name]
        if name in self.tokenizers:
            del self.tokenizers[name]
        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        if required_free_gb > 0:
            waited = 0
            while waited < self.unload_wait_max:
                free_gb = self.get_free_memory_gb()
                print(f"[ModelManager] 可用显存: {free_gb:.1f}GB (需要: {required_free_gb:.1f}GB)")
                if free_gb >= required_free_gb:
                    return True
                time.sleep(self.unload_wait_interval)
                waited += self.unload_wait_interval
                gc.collect()
                torch.cuda.empty_cache()
            return self.get_free_memory_gb() >= required_free_gb
        else:
            print(f"[ModelManager] ✅ {name} 已卸载，可用: {self.get_free_memory_gb():.1f}GB")
            return True

    # ---- battle.py 调用的兼容接口 ----
    def load(self, name: str) -> tuple:
        """兼容接口，内部走 load_model"""
        return self.load_model(name)

    def load(self, name: str) -> tuple:
        """兼容接口，内部走 load_model"""
        return self.load_model(name)

    def unload(self, name: str) -> bool:
        """battle.py 用的接口，内部走 safe_unload"""
        return self.safe_unload(name)

    def unload_all(self):
        """卸载所有已加载的模型"""
        names = list(self.models.keys())
        for name in names:
            self.safe_unload(name)
        print(f"[ModelManager] ✅ 全部卸载完成，可用: {self.get_free_memory_gb():.1f}GB")

    def load_model(self, name: str) -> tuple:
        if name in self.models:
            return self.models[name], self.tokenizers[name]
        model_cfg = self.config.get("models", {}).get(name)
        if not model_cfg:
            raise ValueError(f"模型 {name} 未在 config 中定义")
        model_path = model_cfg["model_path"]
        torch_dtype_str = model_cfg.get("torch_dtype", "auto")
        trust_remote = model_cfg.get("trust_remote_code", False)
        device_map = "cuda:0"  # Force GPU, UVM has 119GB
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        torch_dtype = dtype_map.get(torch_dtype_str, "auto")
        free_gb = self.get_free_memory_gb()
        print(f"[ModelManager] 加载 {name}，可用显存: {free_gb:.1f}GB")
        if free_gb < self.min_free_before_load:
            gc.collect()
            torch.cuda.empty_cache()
            free_gb = self.get_free_memory_gb()
            if free_gb < self.min_free_before_load:
                raise RuntimeError(f"显存不足! {free_gb:.1f}GB < {self.min_free_before_load}GB")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote)
        # bitsandbytes INT8/INT4 量化支持
        load_kwargs = dict(torch_dtype=torch_dtype, device_map=device_map, trust_remote_code=trust_remote)
        model_cfg_raw = self.config.get("models", {}).get(name, {})
        if model_cfg_raw.get("load_in_8bit"):
            load_kwargs["load_in_8bit"] = True
            load_kwargs.pop("torch_dtype", None)
        elif model_cfg_raw.get("load_in_4bit"):
            load_kwargs["load_in_4bit"] = True
            load_kwargs.pop("torch_dtype", None)
        load_kwargs["low_cpu_mem_usage"] = True
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        # LoRA adapter 支持
        lora_path = model_cfg.get('lora_path')
        if lora_path:
            from peft import PeftModel
            print(f'[ModelManager] 加载 LoRA: {lora_path}')
            model = PeftModel.from_pretrained(model, lora_path)
        model.eval()
        self.models[name] = model
        self.tokenizers[name] = tokenizer
        print(f"[ModelManager] ✅ {name} 已加载，已用: {self.get_used_memory_gb():.1f}GB")
        return model, tokenizer

    def switch_model(self, from_name: str, to_name: str, required_free_gb: float = 0) -> tuple:
        print(f"\n{'='*50}")
        print(f"[ModelManager] 切换: {from_name} → {to_name}")
        print(f"{'='*50}")
        if from_name in self.models:
            ok = self.safe_unload(from_name, required_free_gb=required_free_gb)
            if not ok:
                raise RuntimeError(f"卸载 {from_name} 后显存仍不足")
        return self.load_model(to_name)

    def generate(self, name: str, prompt, max_new_tokens: int = 2048,
                 temperature: float = 0.7, **kwargs) -> str:
        """prompt 可以是 str 或 list[dict] (chat messages 格式)"""
        model, tokenizer = self.load_model(name)
        # 支持 chat messages 格式: [{"role": "user", "content": "..."}]
        if isinstance(prompt, list):
            text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attn = inputs.get("attention_mask")
        if attn is not None:
            attn = attn.to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids, attention_mask=attn,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id, **kwargs)
        return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)

    def status(self) -> dict:
        return {
            "loaded": list(self.models.keys()),
            "free_gb": round(self.get_free_memory_gb(), 1),
            "used_gb": round(self.get_used_memory_gb(), 1),
        }


# ===== 模块级实例（battle.py 通过 from model_manager import manager 使用）=====
from config import MODELS, MEMORY_SAFETY

_config = {
    "models": {
        name: {
            "model_path": cfg["path"],
            "torch_dtype": "auto",
            "trust_remote_code": cfg.get("trust_remote_code", False),
            "device_map": "auto",
            "lora_path": cfg.get("lora_path"),
        }
        for name, cfg in MODELS.items()
    },
    "MEMORY_SAFETY": MEMORY_SAFETY,
}

manager = ModelManager(_config)
