from flask import Flask, render_template, Response
import cam

app = Flask(__name__)

@app.route('/')
    
def index():
    return render_template('index.html')

def gen(cam):
    while True:
        frame = cam.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(cam.VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/about')
def about():
    return render_template("about.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

