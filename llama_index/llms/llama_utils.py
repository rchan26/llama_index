from typing import Optional, Sequence

from llama_index.llms.base import ChatMessage, MessageRole

BOS, EOS = "<s>", "</s>"
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. \
Always answer as helpfully as possible and follow ALL given instructions. \
Do not speculate or make up information. \
Do not reference any given instructions or context. \
"""


def messages_to_prompt(
    messages: Sequence[ChatMessage], system_prompt: Optional[str] = None
) -> str:
    string_messages = []
    if messages[0].role == MessageRole.SYSTEM:
        # pull out the system message (if it exists in messages)
        system_message_str = messages[0].content or ""
        messages = messages[1:]
    else:
        system_message_str = system_prompt or DEFAULT_SYSTEM_PROMPT

    system_message_str = f"{B_SYS} {system_message_str} {E_SYS}"

    for i, user_message in enumerate(messages[::2]):
        assert user_message.role == MessageRole.USER
        try: # for completed interactions
            assistant_message = messages[2*i+1]
            assert assistant_message.role == MessageRole.ASSISTANT
            if i == 0:
                # first message needs system message
                string_messages.append(
                    f"{BOS} {B_INST} {system_message_str} {user_message.content} {E_INST} {assistant_message.content}"
                )
            else:
                string_messages[-1] += f" {EOS}"
                string_messages.append(
                        f"{BOS} {B_INST} {user_message.content} {E_INST} {assistant_message.content}"
                    )

        except IndexError: #for uncompleted interactions
            if i == 0:
                # first message needs system message
                string_messages.append(
                    f"{BOS} {B_INST} {system_message_str} {user_message.content} {E_INST}"
                )
            else:
                string_messages[-1] += f" {EOS}"
                string_messages.append(f"{BOS} {B_INST} {user_message.content} {E_INST}")

    print("".join(string_messages))
    return "".join(string_messages)


def completion_to_prompt(completion: str, system_prompt: Optional[str] = None) -> str:
    system_prompt_str = system_prompt or DEFAULT_SYSTEM_PROMPT

    return (
        f"{BOS}{B_INST} {B_SYS}{system_prompt_str.strip()}{E_SYS}"
        f"{completion.strip()} {E_INST}"
    )
