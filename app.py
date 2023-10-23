from flask import Flask, render_template, request
import pickle
# from sklearn import tree
# from sklearn.multioutput import MultiOutputRegressor

app = Flask(__name__)


@app.route('/', methods=['get', 'post'])
def predict():
    message = ''
    if request.method == 'POST':
        i_w = request.form.get('IW')
        i_f = request.form.get('IF')
        v_w = request.form.get('VW')
        f_p = request.form.get('FP')
        
        # print(i_w, i_f)
        
        inp = [[float(i_w), float(i_f), float(v_w), float(f_p)]]

        with open("welding_seam_model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("welding_seam_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        inp = scaler.transform(inp)
        pred = model.predict(inp)
        # print(pred)
        message = f'При заданных параметрах сварки глубина и ширина сварного шва составит {pred[0][0]}, {pred[0][1]}'
        print(message)

    return render_template('index.html', message=message)


@app.route('/text/')
def print_text():
    return '<h1>Some text</h1'


# app.run()
