import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras_hub
import keras
import tensorflow as tf
import time

keras.mixed_precision.set_global_policy("mixed_float16")

preprocessor = keras_hub.models.GPT2CausalLMPreprocessor.from_preset(
    "gpt2_base_en",
    sequence_length=1024,
)
gpt2_lm = keras_hub.models.GPT2CausalLM.from_preset(
    "gpt2_base_en", preprocessor=preprocessor
)

start = time.time()

output = gpt2_lm.generate("Hi I am a boy and I am a professor and I do", max_length=50)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")