import torch
from chronos_forecasting.src.chronos.chronos import ChronosPipeline
from chronos_forecasting.src.chronos.chronos_bolt import ChronosBoltPipeline

class ChronosRepr:
    def __init__(self, model_name):
        if model_name == "chronos":
            self.pipeline = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-small",
                device_map="cuda",
                torch_dtype=torch.bfloat16,
            )
        elif model_name == "chronos_bolt":
            self.pipeline = ChronosBoltPipeline.from_pretrained(
                "amazon/chronos-bolt-small",
                device_map="cuda",
                torch_dtype=torch.bfloat16,
            )  
        else:
            raise ValueError(f"Invalid model name: {model_name}")

    def get_repr(self, context):
        embeddings, _ = self.pipeline.embed(context)
        return embeddings
