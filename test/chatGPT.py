import openai

openai.api_key = "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"

model_engine = "text-davinci-003"
prompt = "what is the capital of France?"

completion = openai.Completion.create(
    engine = model_engine,
    prompt = prompt,
    max_tokens = 1024,
    n = 1,
    stop = None,
    temperature = 0.5,
)
message = completion.choices[0].text

print(message)