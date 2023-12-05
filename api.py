import json

import requests
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import pandas as pd
import re

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-squad")
model = AutoModelForQuestionAnswering.from_pretrained("savasy/bert-base-turkish-squad")
nlp = pipeline("question-answering", model=model, tokenizer=tokenizer)

# pandas ile ilgili islemler

df = pd.read_json("data.json")
bdf = pd.read_json("body.json")
cdf = pd.read_json("color.json")


def find_matching_color(color):
    colorAnswer = color.lower()
    colorReplace = colorAnswer.replace(',', '')
    colors = colorReplace.split()

    filter_data_color = cdf[
        cdf['colors'].isin(colors)
    ]

    return filter_data_color["colors"].tolist()


def find_matching_size(sizes):
    sizeAnswer = sizes.lower()
    sizes = sizeAnswer.split()

    filter_data = bdf[
        bdf['sizes'].isin(sizes)
    ]

    return filter_data["sizes"].tolist()


def find_matching_wear(kind):
    kindAnswer = kind.lower()
    colorReplace = kindAnswer.replace(',', '')
    kinds = colorReplace.split()

    return kinds


def find_matching_data(kind, sizes, price, color):
    filtered_data = df[
        (df['kind'].isin(kind)) &
        (df['body'].isin(sizes)) &
        (df['price'] <= price) &
        (df['color'].isin(color))
        ]

    if not filtered_data.any().any():
        filtered_data_special = df[
            (~df['kind'].isin(['fular', 'atkı'])) &
            (df['price'] <= price) &
            (df['color'].isin(color))
            ]

        return filtered_data_special

    return filtered_data


def chose_random_product(color, price):
    filtered_data = df[
        (df['kind'].isin(['fular', 'atkı'])) &
        (df['price'] <= price) &
        (df['color'].isin(color))
        ]

    return filtered_data


def find_max_budget(budget):
    numbers = [int(match.group()) for match in re.finditer(r'\d+',
                                                           budget)]  # metni işleyerek sayısal değeri ayıkladım. Python'da düzenli ifadeleri (regex) kullanarak bunu gerçkeltirebiliriz
    maxNumber = max(numbers)

    return maxNumber


@app.route('/message', methods=['POST'])
def process_message():
    global result_content
    data = request.get_json()

    received_message = data.get('message', '')
    print("gelen mesaj : ", received_message)

    answerCombine = nlp(question="nasıl bir kombin olutşurmak istiyor?", context=received_message)

    answerWear = nlp(question="neler giymek istiyor? veya ne tür giyinmek istiyor?", context=received_message)

    answerColor = nlp(
        question="hangi renk giymek istiyor veya en sevdiği renk ne? veya bu metinde hangi renklerden bahsediyor?",
        context=received_message)

    answerBudget = nlp(question="ne kadarlık parası var? veya ne kadarlık bütcesi var?", context=received_message)


    answerBody = nlp(question="hangi beden harfini tercih ediyor?", context=received_message)
    print("combine : ", answerCombine["answer"])
    print("kind : ", find_matching_wear(answerWear["answer"]))
    print("size : ", find_matching_size(answerBody["answer"]))
    print("price : ", find_max_budget(answerBudget["answer"]))
    print("color : ", find_matching_color(answerColor["answer"]))

    kind = find_matching_wear(answerWear["answer"])  # i found kind
    body = find_matching_size(answerBody["answer"])  # i found clothes size
    price = find_max_budget(answerBudget["answer"])  # i found max price
    color = find_matching_color(answerColor["answer"])  # i found colors

    matching_data = find_matching_data(kind, body, price, color)
    print("data : ", matching_data)
    x = chose_random_product(color, price)
    print("alternative: ", x)

    result_dict = matching_data.to_dict(orient='records')

    messages = f"bir kombin oluşturmak istiyorum. bütçem {price}. bedenim {body}. sevdiğim renkler {color}. kombini şu verilerden yap : {result_dict} ve bana yaptığın dataların ver maximum 3 kombin yap"

    json_data = {
        "message": messages
    }

    response = requests.post("http://localhost:8888/message", json=json_data)

    if response.status_code == 200:
        try:
            if response.content:
                print("Raw Response Content:", response.content)

                result = response.json()
                print("Result:", result)

                result_content = result.get('result')
                print("Result Content:", result_content)

                print(result_content)
            else:
                print('Response content is empty.')
        except json.JSONDecodeError as e:
            print('Error decoding JSON:', e)
    else:
        print('Request failed:', response.status_code)

    return jsonify({'response': result_content})


@app.route('/ping', methods=['GET'])
def ping():
    return 'Ping!'


if __name__ == '__main__':
    # Uygulamayı 5000 portunda çalıştır
    app.run(port=5000)
