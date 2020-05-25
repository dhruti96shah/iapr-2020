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
    parser.add_argument('--input', default=None, type=str, help='input avi video directory')
    parser.add_argument('--output', default=None, type=str, help='output avi video directory')
    args = parser.parse_args()
    return args


def main(args):
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'MJPG'), 2, (720, 480), True)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # arrow_locations = tip_tracking_2()
    cap = cv2.VideoCapture(args.input)
    t = 0
    eqn = ""
    dict_op = {'plus':'+','div':'/','eq':'=','mult':'*','minus':'-'}
    res = 0
    digit_or_op = True
    _traj_arrow_tip_locations = []
    _traj_k = 0
    while(cap.isOpened()):
        print('processing frame: {} '.format(t+1))
        ret, _frame = cap.read()
        if ret==False: #if video is over
            break
        if t==0: #save the first frame
            first_frame = _frame

        _traj_k+=1
        arrow_loc = tip_tracking_3(_frame)
        _traj_arrow_tip_locations.append(arrow_loc)
        # draw linear trajectory
        if _traj_k > 1:
            # start_point = tuple(np.flip(arrow_tip_locations[-1]))
            # end_point = tuple(np.flip(arrow_tip_locations[-2]))
            # Black color in BGR
            color = (0, 100, 255)
            # Line thickness of 5 px
            thickness = 5
            for kk in range(_traj_k, 1, -1):
                _frame = cv2.line(_frame, tuple(np.flip(_traj_arrow_tip_locations[-kk + 1])),
                                  tuple(np.flip(_traj_arrow_tip_locations[-kk])), color, thickness)

        cv2.putText(_frame,"Equation: " + eqn,(20,400), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        writer.write(_frame)
        # cv2.imshow('frame',_frame)
        # cv2.waitKey(1000)

        # Get the area around the arrow tip, from which the image is to be detected
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
                        prediction, loss = pred_digit(torch.tensor(im_for_digit/255).float().unsqueeze(0).unsqueeze(0),False)
                        if stats[4]>90:
                            # cv2.imshow('frame',im_for_digit)
                            # cv2.waitKey(1000)
                            digit_or_op = False
                            eqn += " " + str(prediction)
                            # cv2.putText(_frame,eqn,(20,400), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                            # cv2.imshow('frame',_frame)
                            print("Predicted digit: ",prediction," Loss: ",loss)

        # If operator to detect
        else:
            op, loss = operator_classify(img)
            if loss<1.5e-10:
                digit_or_op = True
                if op!='eq':
                    eqn += " " + str(dict_op[op])
                print("Predicted operator: ",op)
                # Skip the next two frames, so as not to interfere with digit classification
                # cv2.imwrite('/home/mahdi/IAPR/iapr-2020/tmp/final_frames/frame_{}.png'.format(t), _frame)
                t+=1
                ret, _frame = cap.read()

                _traj_k += 1
                arrow_loc = tip_tracking_3(_frame)
                _traj_arrow_tip_locations.append(arrow_loc)
                # draw linear trajectory
                if _traj_k > 1:
                    # start_point = tuple(np.flip(arrow_tip_locations[-1]))
                    # end_point = tuple(np.flip(arrow_tip_locations[-2]))
                    # Black color in BGR
                    color = (0, 100, 255)
                    # Line thickness of 5 px
                    thickness = 5
                    for kk in range(_traj_k, 1, -1):
                        _frame = cv2.line(_frame, tuple(np.flip(_traj_arrow_tip_locations[-kk + 1])),
                                          tuple(np.flip(_traj_arrow_tip_locations[-kk])), color, thickness)

                cv2.putText(_frame,"Equation: " + eqn,(20,400), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                writer.write(_frame)

                # cv2.imshow('frame',_frame)
                # cv2.waitKey(1000)
                # cv2.imwrite('/home/mahdi/IAPR/iapr-2020/tmp/final_frames/frame_{}.png'.format(t), _frame)
                t+=1
                ret, _frame = cap.read()

                _traj_k += 1
                arrow_loc = tip_tracking_3(_frame)
                _traj_arrow_tip_locations.append(arrow_loc)
                # draw linear trajectory
                if _traj_k > 1:
                    # start_point = tuple(np.flip(arrow_tip_locations[-1]))
                    # end_point = tuple(np.flip(arrow_tip_locations[-2]))
                    # Black color in BGR
                    color = (0, 100, 255)
                    # Line thickness of 5 px
                    thickness = 5
                    for kk in range(_traj_k, 1, -1):
                        _frame = cv2.line(_frame, tuple(np.flip(_traj_arrow_tip_locations[-kk + 1])),
                                          tuple(np.flip(_traj_arrow_tip_locations[-kk])), color, thickness)

                cv2.putText(_frame, "Equation: " + eqn, (20, 400), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                writer.write(_frame)

                # cv2.imshow('frame',_frame)
                # cv2.waitKey(1000)
                if op=='eq':
                    res = evaluate(eqn)
                    eqn += " " + str(dict_op[op])
                    cv2.putText(_frame,"Equation: " + eqn + " " + str(res),(20,400), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    writer.write(_frame)
                    print(eqn + " " + str(res))
                    # cv2.imshow('frame',_frame)
                    # cv2.waitKey(100)
                    # cv2.imwrite('/home/mahdi/IAPR/iapr-2020/tmp/final_frames/frame_{}.png'.format(t), _frame)

                    while True:  # if video is over breaks
                        t += 1
                        ret, _frame = cap.read()
                        if ret==False:
                            break
                        _traj_k += 1
                        arrow_loc = tip_tracking_3(_frame)
                        _traj_arrow_tip_locations.append(arrow_loc)
                        # draw linear trajectory
                        if _traj_k > 1:
                            # start_point = tuple(np.flip(arrow_tip_locations[-1]))
                            # end_point = tuple(np.flip(arrow_tip_locations[-2]))
                            # Black color in BGR
                            color = (0, 100, 255)
                            # Line thickness of 5 px
                            thickness = 5
                            for kk in range(_traj_k, 1, -1):
                                _frame = cv2.line(_frame, tuple(np.flip(_traj_arrow_tip_locations[-kk + 1])),
                                                  tuple(np.flip(_traj_arrow_tip_locations[-kk])), color, thickness)
                        cv2.putText(_frame, "Equation: " + eqn + " " + str(res), (20, 400), font, 1, (0, 0, 0), 2,
                                    cv2.LINE_AA)
                        # cv2.imwrite('/home/mahdi/IAPR/iapr-2020/tmp/final_frames/frame_{}.png'.format(t), _frame)
                        writer.write(_frame)

        if ret==False:
            break
        # cv2.imwrite('/home/mahdi/IAPR/iapr-2020/tmp/final_frames/frame_{}.png'.format(t), _frame)
        t+=1
    pass


if __name__ == '__main__':
    args = cli()
    # args.input = '/home/mahdi/IAPR/iapr-2020/tmp/input_dir/robot_parcours_1.avi'
    # args.output = '/home/mahdi/IAPR/iapr-2020/tmp/output_dir/result.avi'
    main(args)
