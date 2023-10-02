## List of all task, dataset, and prompt names:
ALL_TASKS = [
    {
        "hf_identifier" : "ShrinivasSK/small-hi-kn",
        "prompt_template" : f"Translate Hindi to Kannada:\n {{input}}\nTranslation:\n",
    },
    {
        "hf_identifier" : "ShrinivasSK/small-hi-te",
        "prompt_template" : f"Translate Hindi to Telugu:\n {{input}}\nTranslation:\n",
    },
    {
        "hf_identifier" : "ShrinivasSK/small-hi-kn2",
        "prompt_template" : f"Translate Hindi to Telugu:\n {{input}}\nTranslation:\n",
    }
]

TASK_TO_PROMPT_TEMPLATE = {}
for task in ALL_TASKS:
    TASK_TO_PROMPT_TEMPLATE[task["hf_identifier"]] = task["prompt_template"]
