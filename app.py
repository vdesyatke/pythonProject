from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['get', 'post'])
def predict():
    if request.method == 'POST':
        i_w = request.form.get('IW')
        i_f = request.form.get('IF')
        v_w = request.form.get('VW')
        f_p = request.form.get('FP')

        inp = np.array([[float(i_w),float(i_f), float(v_w), float(f_p)]])

        with open("/models/welding_seam_model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("/models/welding_seam_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        inp = scaler.transform(inp)
        predict = model.predict(inp)

    return render_template('index.html')


@app.route('/text/')
def print_text():
    return '<h1>Some text</h1'

