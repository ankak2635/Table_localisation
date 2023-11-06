import os

import sys
sys.path.append('/home/ankit/Data_Science/CV_Projects/OrderStack')  # fix import errors

import tornado.ioloop
import tornado.web
import cv2
from tornado import gen
import base64
import numpy as np

from src.inference.predict import Predict


class InferenceHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("app_web_page.html")

    @gen.coroutine
    def post(self):
        try:
            # Get the uploaded image file
            uploaded_file = self.request.files['image'][0]
            image_data = uploaded_file['body']

            # Perform inference on the uploaded image
            predicted_image = self.run_inference(image_data)

            # Display the predicted image
            self.write('<h1>Predicted Image</h1>')
            self.write('<img src="data:image/png;base64,{}"/>'.format(predicted_image))

        except Exception as e:
            self.write(f'<h1>Error</h1><p>An error occurred: {str(e)}</p>')

    def run_inference(self, image_data):
        try:
            # Convert the image data to cv2 format
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

            # make inference
            prediction_obj= Predict()
            predictions= prediction_obj.new_image_predict(image=image)

            # Encode the image as bytes
            _, buffer = cv2.imencode(".png", predictions)
            image_bytes = buffer.tobytes()

            # Encode the image bytes as base64
            predicted_image = base64.b64encode(image_bytes)
            
            return predicted_image
        
        except Exception as e:
            raise e
        
def make_app():
    return tornado.web.Application([
        (r"/", InferenceHandler),
    ], debug=True, autoreload=True, static_path=os.path.join(os.path.dirname(__file__), "static"))
    

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    print("Tornado server is running at http://localhost:8888")
    tornado.ioloop.IOLoop.current().start()

