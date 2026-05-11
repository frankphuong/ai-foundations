# This is begin of the process to learning become AI Engineer (topic LLM call).

In the first place, I have a task from my boss: Our company need an AI agent can classify request from user (email, phone call, direct message).
So, basically, I am just create a call from client and sent all user's message to LLM to process, and let's LLM know what to do: classify this user's message
as some category: billing, account, payment, ...

Ok now let's deep dive in the file `b01_basic.py` with class `BasicClassifyAgent`

```python

class BasicUserRequestClassificationAgent(BaseModel):
```
