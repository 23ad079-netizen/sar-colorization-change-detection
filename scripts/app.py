# import os
# import io
# import numpy as np
# from flask import Flask, render_template, request, send_file
# from werkzeug.utils import secure_filename
# from PIL import Image

# from utils import colorize_array, detect_changes_color

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# @app.route('/')
# def index():
#     return render_template('index.html')


# # --- route 1: colorize uploaded SAR image ---
# @app.route('/colorize', methods=['POST'])
# def colorize_route():
#     if 'file' not in request.files:
#         return 'No file uploaded', 400

#     f = request.files['file']

#     # read image as grayscale, resize to 256x256, normalize to [0,1]
#     img = Image.open(f).convert('L').resize((256, 256))
#     sar = np.array(img, dtype=np.float32) / 255.0   # (H, W)

#     # use model on array
#     color = colorize_array(sar)                     # (H, W, 3) in [0,1]
#     img_out = Image.fromarray((color * 255).astype('uint8'))

#     buf = io.BytesIO()
#     img_out.save(buf, format='PNG')
#     buf.seek(0)
#     return send_file(buf, mimetype='image/png')


# # --- route 2: change detection on two uploaded COLOR images ---
# @app.route('/change-detect', methods=['POST'])
# def change_route():
#     if 'file1' not in request.files or 'file2' not in request.files:
#         return 'Two files required', 400

#     f1 = request.files['file1']
#     f2 = request.files['file2']

#     # read both as RGB, resize, normalize to [0,1]
#     img1 = Image.open(f1).convert('RGB').resize((256, 256))
#     img2 = Image.open(f2).convert('RGB').resize((256, 256))

#     color1 = np.array(img1, dtype=np.float32) / 255.0   # (H, W, 3)
#     color2 = np.array(img2, dtype=np.float32) / 255.0

#     overlay = detect_changes_color(color1, color2)      # (H, W, 3) in [0,1]

#     img_out = Image.fromarray((overlay * 255).astype('uint8'))
#     buf = io.BytesIO()
#     img_out.save(buf, format='PNG')
#     buf.seek(0)
#     return send_file(buf, mimetype='image/png')


# if __name__ == '__main__':
#     app.run(debug=True)








import os
import io
import numpy as np
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
from PIL import Image

from utils import colorize_array, detect_changes_color

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


# --- route 1: colorize uploaded SAR image ---
@app.route('/colorize', methods=['POST'])
def colorize_route():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    f = request.files['file']
    if f.filename == '':
        return 'No file selected', 400

    # get terrain from form (sent by index.html)
    terrain = request.form.get('terrain', 'barrenland').lower()

    # read image as grayscale, resize to 256x256, normalize to [0,1]
    img = Image.open(f).convert('L').resize((256, 256))
    sar = np.array(img, dtype=np.float32) / 255.0   # (H, W)

    # use terrain-specific model on array
    color = colorize_array(sar, terrain=terrain)    # (H, W, 3) in [0,1]
    img_out = Image.fromarray((color * 255).clip(0, 255).astype('uint8'))

    buf = io.BytesIO()
    img_out.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


# --- route 2: change detection on two uploaded COLOR images ---
@app.route('/change-detect', methods=['POST'])
def change_route():
    if 'file1' not in request.files or 'file2' not in request.files:
        return 'Two files required', 400

    f1 = request.files['file1']
    f2 = request.files['file2']

    # read both as RGB, resize, normalize to [0,1]
    img1 = Image.open(f1).convert('RGB').resize((256, 256))
    img2 = Image.open(f2).convert('RGB').resize((256, 256))

    color1 = np.array(img1, dtype=np.float32) / 255.0   # (H, W, 3)
    color2 = np.array(img2, dtype=np.float32) / 255.0

    overlay = detect_changes_color(color1, color2)      # (H, W, 3) in [0,1]

    img_out = Image.fromarray((overlay * 255).clip(0, 255).astype('uint8'))
    buf = io.BytesIO()
    img_out.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
