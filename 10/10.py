import openai # type: ignore

# 問答迴圈
while True:
    question = input("請輸入問題（輸入 'exit' 離開）：")
    if question.lower() == "exit":
        break

    # 呼叫 GPT 模型
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # 或 "gpt-4"（需有權限）
        messages=[
            {"role": "system", "content": "你是一個有幫助的問答助手。"},
            {"role": "user", "content": question}
        ]
    )

    # 印出回答
    answer = response["choices"][0]["message"]["content"]
    print("GPT 回答：", answer)
