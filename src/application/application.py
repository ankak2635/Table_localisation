import os

import sys
sys.path.append('/home/ankit/Data_Science/CV_Projects/OrderStack')  # fix import errors

import tornado.ioloop
import tornado.web
import cv2
from tornado import gen
import numpy as np
from PIL import Image

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
            predicted_image_filename = self.run_inference(image_data)

            # Display the predicted image
            self.write('<h1>Predicted Image</h1>')
            self.write(f'<img src="/static/{predicted_image_filename}" width="auto" height="auto" style="max-width:100%; max-height:100%;" />')

        except Exception as e:
            self.write(f'<h1>Error</h1><p>An error occurred: {str(e)}</p>')

    def run_inference(self, image_data):
        try:
            # Convert the image data to cv2 format
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

            # Make inference
            prediction_obj = Predict()
            predicted_image = prediction_obj.new_image_predict(image=image)

            # Define the filename to save the predicted image in the "static" folder
            predicted_image_filename = "predicted_image.jpg"
            predicted_image_path = os.path.join(os.path.dirname(__file__), "static", predicted_image_filename)

            # Save the predicted image to the "static" folder
            cv2.imwrite(predicted_image_path, predicted_image)

            return predicted_image_filename

        except Exception as e:
            raise e

def make_app():
    return tornado.web.Application([
        (r"/", InferenceHandler),
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": os.path.join(os.path.dirname(__file__), "static")}),
    ], debug=True, autoreload=True)

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    print("Tornado server is running at http://localhost:8888")
    tornado.ioloop.IOLoop.current().start()
