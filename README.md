# LLM-PPL

An example showing how to calculate the perplexity for LLaMA.

If you want to print the PPL of the current training step in the log, you can do it as follows：
```python
import math
from transformers import Trainer
from transformers.trainer_callback import TrainerCallback

class MyTrainerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs, **kwargs):
        # compute PPL metrics and call back while printing log
        logs["PPL"] = math.exp(logs["loss"])

trainer = Trainer(callbacks=[MyTrainerCallback])
```
