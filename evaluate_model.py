from transformers import AutoTokenizer, AutoModelForCausalLM, BertForMaskedLM
import json


def bert_model():
    model_name = "bert-base-uncased"
    # model_name = "./bert-finetuned-squad/checkpoint-33276"
    model = BertForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def llama_model():
    # model_name = "openbmb/MiniCPM-Llama3-V-2_5"
    # model_name = "meta-llama/Meta-Llama-3-8B"
    model_name = "nvidia/Llama3-ChatQA-1.5-8B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def answer_prompt(model_tokenizer, input_prompt: str) -> str:
    model = model_tokenizer[0]
    tokenizer = model_tokenizer[1]

    inputs = tokenizer.encode(
        # for LLAMA
        tokenizer.bos_token + input_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        # for Bert
        # input_prompt, return_tensors="pt", truncation=True, max_length=512
    ).to(model.device)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    outputs = model.generate(
        input_ids=inputs,
        # input_ids=inputs.input_ids,
        # attention_mask=inputs.attention_mask,
        max_new_tokens=128,
        # eos_token_id=terminators,
    )
    # response = outputs[0][inputs.input_ids.shape[-1]:]
    response = outputs[0]  # [inputs.shape[-1]:]
    answer = tokenizer.decode(response, skip_special_tokens=True)
    answer = answer[len(input_prompt) :]  # strip away prompt

    # answer_start_scores, answer_end_scores = model(**inputs).values()

    # answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
    # answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

    # answer = tokenizer.convert_tokens_to_string(
    #     tokenizer.convert_ids_to_tokens(
    #         inputs['input_ids'][answer_start:answer_end]
    #         )
    # )

    return answer


def main():
    model = llama_model()
    with open("./data/SQuAD/dev-v2.0.json") as f:
        input_json = json.load(f)
        output_data = {}
        i = 0
        for data in input_json["data"]:
            for paragraph in data["paragraphs"]:
                for qas in paragraph["qas"]:
                    input_prompt = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context.\n\n"
                    input_prompt += paragraph["context"] + "\n\n"
                    input_prompt += "User: " + qas["question"] + "\n\n"
                    input_prompt += "Assistant: "
                    answer = answer_prompt(model, input_prompt)
                    output_data[qas["id"]] = answer
                    print(answer)
                i += 1
                with open(f"predictions-dev-squad_llama_gpu_step-{i}.json", "w") as o:
                    json.dump(output_data, o)


if __name__ == "__main__":
    main()
