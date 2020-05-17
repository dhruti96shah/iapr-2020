import argparse
from localization import *
from digit_classification import *
from operator_classification import *
import numpy as np
from equation_evaluation import *

import warnings

warnings.simplefilter("ignore", UserWarning)



def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--vis', default=True,
                        help='visualization')

def main(args):
    font = cv2.FONT_HERSHEY_SIMPLEX
    network = Net()
    network_state_dict = torch.load('./model.pth')
    network.load_state_dict(network_state_dict)
    network.eval()
    

    arrow_locations = tip_tracking()
    cap = cv2.VideoCapture('robot_parcours_1.avi')
    t = 0
    eqn = ""
    eqn_list = []
    dict_op = {'plus':'+','div':'/','eq':'=','mult':'*','minus':'-'}
    res = 0
    digit_or_op = True

    while(cap.isOpened()):
        ret, _frame = cap.read()
        if ret==False: #if video is over
            break
        if t==0: #save the first frame
            first_frame = _frame
            
        cv2.putText(_frame,"Equation: " + eqn,(20,400), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame',_frame)
        cv2.waitKey(1000)

        # Get the area around the arrow tip, from which the image is to be detected
        arrow_loc = arrow_locations[t]
        x_min = max(arrow_loc[0]-40, 0)
        x_max = min(arrow_loc[0]+40, first_frame.shape[0])
        y_min = max(arrow_loc[1]-40,0)
        y_max = min(arrow_loc[1]+40, first_frame.shape[1])
        img = first_frame[x_min:x_max,y_min:y_max]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # If digit to detect
        if digit_or_op:

            #Threshold the image
            _,im_for_digit = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)

            # Get the bounding box of the digit
            output = cv2.connectedComponentsWithStats(im_for_digit, 8, cv2.CV_32S)
            stats = output[2]
            if output[2].shape[0]==2: #If only one connected component in the image
                stats = stats[0] 
                if(stats[4]>0): # Area of connected component>0

                    # Get image ready for passing to cnn model ie get 28*28 image
                    im_for_digit = im_for_digit[stats[1]:stats[1]+stats[3],stats[0]:stats[0]+stats[2]]
                    delta_w = 28 - im_for_digit.shape[1]
                    delta_h = 28 - im_for_digit.shape[0]
                    top, bottom = delta_h//2, delta_h-(delta_h//2)
                    left, right = delta_w//2, delta_w-(delta_w//2)
                    if top>0 and bottom>0 and left>0 and right>0:
                        color = [255]
                        im_for_digit = cv2.copyMakeBorder(im_for_digit, top, bottom, left, right, cv2.BORDER_CONSTANT,
                            value=color)
                        im_for_digit = cv2.bitwise_not(im_for_digit)
                        im_for_digit = cv2.morphologyEx(im_for_digit, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
                        # cv2.imshow('frame',im_for_digit)
                        # cv2.waitKey(1000)
                        prediction, loss = pred_digit(torch.tensor(im_for_digit/255).float().unsqueeze(0).unsqueeze(0), network)
                        if stats[4]>90:
                            # cv2.imshow('frame',im_for_digit)
                            # cv2.waitKey(1000)
                            digit_or_op = False
                            eqn += " " + str(prediction)
                            eqn_list.append(int(prediction))
                            cv2.putText(_frame,eqn,(20,400), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                            cv2.imshow('frame',_frame)
                            print("Predicted digit: ",prediction)#," Loss: ",loss)

        # If operator to detect
        else:
            op, loss = operator_classify(img)
            if loss<1.5e-10:
                digit_or_op = True
                if op!='eq':
                    eqn += " " + str(dict_op[op])
                    eqn_list.append(str(dict_op[op]))
                print("Predicted operator: ",op)
                # Skip the next two frames, so as not to interfere with digit classification
                t+=1
                ret, _frame = cap.read()
                cv2.putText(_frame,"Equation: " + eqn,(20,400), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow('frame',_frame)
                cv2.waitKey(1000)
                t+=1
                ret, _frame = cap.read()
                cv2.putText(_frame,"Equation: " + eqn,(20,400), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow('frame',_frame)
                cv2.waitKey(1000)
                if op=='eq':
                    res = evaluate(eqn)
                    eqn += " " + str(dict_op[op])
                    cv2.putText(_frame,"Equation: " + eqn + " " + str(res),(20,400), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    print(eqn + " " + str(res))
                    cv2.imshow('frame',_frame)
                    cv2.waitKey(1000)
                    break
        
        t+=1
    pass


if __name__ == '__main__':
    args = cli()
    main(args)