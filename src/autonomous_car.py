import cv2
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import modi

class AutonomousCar(object):

    def __init__(self):
        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)

        # Load the model
        self.model = tensorflow.keras.models.load_model('../model/keras_model.h5')

        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1.
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


    def start_car(self,mot):

        camera = cv2.VideoCapture(0)
        camera.set(3, 224)
        camera.set(4, 224)
        camera.set(5, 60)

        while camera.isOpened():
            _, frame = camera.read()
            cv2.imshow("frame", frame)

            # Replace this with the path to your image
            # image = Image.open('/Users/peter/Repos/ai-curriculum-autonomous-car/car_hun/21.jpg')
            image = Image.fromarray(frame)

            # resize the image to a 224x224 with the same strategy as in TM2:
            # resizing the image to be at least 224x224 and then cropping from the center
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.ANTIALIAS)

            # turn the image into a numpy array
            image_array = np.asarray(image)

            # display the resized image
            # image.show()

            # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

            # Load the image into the array
            self.data[0] = normalized_image_array

            # run the inference
            prediction = self.model.predict(self.data)
            pred = self.model.predict_classes(self.data)
            print(prediction)
            print(pred)

            mot.speed = 30,30






            if cv2.waitKey(1) & 0xFF == ord('q'):
                mot.speed = 0,0
                camera.release()
                cv2.destroyAllWindows()
                break


def main():
    car = AutonomousCar()
    bundle = modi.MODI()
    mot = bundle.motors[0]

    car.start_car(mot)




if __name__ == "__main__":
    main()