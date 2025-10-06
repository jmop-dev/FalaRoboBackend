from openai import OpenAI
import os
import json
import requests
from flask import Flask, request
import jsonify

client = OpenAI(api_key='OPENAPI')

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hack the planet!'
    
@app.route('/ai')
def ai():
    question = request.args.get("question")
    return ask(question)



def ask(question):
    try:

        data = requests.get('http://localhost:8000/ai/snapshot').json()
        print(data)

        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You're a chat assistant that is tasked to report on the functionality of a pneumatic system that controls 2 actuators One called Big and the other called Small. You should explain it in simple terms and describe the state of the machine. You must answer in brazilian portuguese."},
                {"role": "user", "content": question},
                {"role": "assistant",
                "content": "Being 1 > bar < 3 the normal pressure of the machine, Here is the data of the machine: " + str(data)},
            ]
        )

        return response.choices[0].message.content, 200

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    app.run(debug=True)