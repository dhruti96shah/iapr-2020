import argparse
from localization import *
from cnn import *
from operator_classification import *
import numpy as np



def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--vis', default=True,
                        help='visualization')

def main(args):
    arrow_locations = tip_tracking()
    cap = cv2.VideoCapture('robot_parcours_1.avi')
    t = 0
    eqn = ""
    digit_or_op = True

    while(cap.isOpened()):
        ret, _frame = cap.read()
        if ret==False: #if video is over
            break
        if t==0: #save the first frame
            first_frame = _frame
        arrow_loc = arrow_locations[t]
        x_min = max(arrow_loc[0]-40, 0)
        x_max = min(arrow_loc[0]+40, first_frame.shape[0])
        y_min = max(arrow_loc[1]-40,0)
        y_max = min(arrow_loc[1]+40, first_frame.shape[1])
        img = first_frame[x_min:x_max,y_min:y_max]
        #if digit_or_op:
            #pass image to cnn and get the prediction
            #append to equation
            # if digit detected, set digit_or_op = False
        #else:
            #pass image to op classifier
            # append to equation
            # if operator detected set digit_or_op = True
            # if operator == 'eq': break
        training_flag = True
        prediction = pred_digit(img,training_flag)
        print("Frame:"+ str(t) + " Predicted digit:"+ str(prediction))
        break
        op, loss = operator_classify(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        if loss<1.5e-10:
            print(op,' ' ,loss)

#         cv2.imshow('frame',img)
#         cv2.waitKey(1000)
        t+=1
    pass


if __name__ == '__main__':
    args = cli()
    main(args)