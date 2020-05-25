from skimage import io
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.morphology import disk, closing
from skimage.color import rgb2hsv, hsv2rgb



def tip_tracking():

    arrow_tip_locations = []
    cap = cv2.VideoCapture('robot_parcours_1.avi')

    k = 0
    while(cap.isOpened()):
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        ret, _frame = cap.read()
        if ret==False:
            break
        k += 1
        frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB)
        frame2=frame.astype('float')

        frame3 = (frame2[:, :, 0] - frame2[:, :, 1]) + (frame2[:, :, 0] - frame2[:, :, 2])
        argoman=np.transpose(np.asarray(np.unravel_index(np.argsort(frame3, axis=None)[::-1], frame3.shape))[:, :2000])

        #  remove pixel outliers
        mean = np.mean(argoman, axis=0)
        standard_deviation = np.std(argoman, axis=0)
        distance_from_mean = abs(argoman - mean)
        max_deviations = 2
        not_color_outlier = np.logical_and(distance_from_mean[:,0] < max_deviations * standard_deviation[0], distance_from_mean[:,1] < max_deviations * standard_deviation[1])
        argoman = argoman[not_color_outlier]



        frame3 *= 0
        frame3[argoman[:, 0], argoman[:, 1]] = 1
        #fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        #ax.imshow(frame3, cmap='gray')

        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        pca.fit(argoman)
        arrow = pca.transform(argoman)
        arg_tip_arrow = argoman[np.argmin(arrow), :]
        arrow_tip_locations.append(arg_tip_arrow)
        # frame3[arg_tip_arrow[0], arg_tip_arrow[1]]=
        #ax.scatter(arg_tip_arrow[1], arg_tip_arrow[0], marker="*", s=100, c='lime')

        #plt.show()


        # Write some Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (5, 25)
        fontScale = 0.5
        fontColor = (255, 255, 255)
        lineType = 1
        cv2.putText(_frame, 'frame # = {}; tip arrow location = {}'.format(k, arg_tip_arrow),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        #cv2.imshow('frame', _frame)

    return arrow_tip_locations



def tip_tracking_2():

    arrow_tip_locations = []
    cap = cv2.VideoCapture('/home/mahdi/IAPR/iapr-2020/data/project_data/robot_parcours_1.avi')

    k = 0
    avg_brightness = []
    while(cap.isOpened()):
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        ret, _frame = cap.read()
        if ret==False:
            break
        k += 1
        frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB)
        def illumination_invariance(im):
            hsv_im = rgb2hsv(im)
            temp = np.ones(hsv_im[:, :, 2].shape)
            # 0.5958 is the mean of brightness for train video frames
            hsv_im[:, :, 2] = np.minimum(hsv_im[:, :, 2] + (0.5958 - np.mean(hsv_im[:, :, 2])), temp)
            avg_brightness.append(np.mean(hsv_im[:, :, 2]))
            return hsv2rgb(hsv_im)
        fig, ax = plt.subplots()
        frame = illumination_invariance(frame)
        ax.imshow((frame))
        # ax.axis('off')
        # plt.show(block=True)

        frame2=frame.astype('float')

        frame3 = (frame2[:, :, 0] - frame2[:, :, 1]) + (frame2[:, :, 0] - frame2[:, :, 2])
        argoman=np.transpose(np.asarray(np.unravel_index(np.argsort(frame3, axis=None)[::-1], frame3.shape))[:, :2000]) #pick 2000 points

        # #  remove pixel outliers
        # mean = np.mean(argoman, axis=0)
        # standard_deviation = np.std(argoman, axis=0)
        # distance_from_mean = abs(argoman - mean)
        # max_deviations = 2
        # not_color_outlier = np.logical_and(distance_from_mean[:,0] < max_deviations * standard_deviation[0], distance_from_mean[:,1] < max_deviations * standard_deviation[1])
        # argoman = argoman[not_color_outlier]


        frame3 *= 0
        frame3[argoman[:, 0], argoman[:, 1]] = 1
        # from skimage.morphology import disk
        # from skimage.filters import median
        # frame3 = median(frame3, disk(3))
        # used openning to remove small islands and find corresponding argoman and remove islands there as well
        # from skimage.morphology import disk, opening
        # frame4 = opening(frame3, disk(3))
        frame4 = frame3
        args_opened = np.argwhere(frame4 != frame3)
        np.in1d(argoman[:, 0], args_opened[:, 0])
        argoman_opened = np.logical_and(np.in1d(argoman[:, 0], args_opened[:, 0]), np.in1d(argoman[:, 1], args_opened[:, 1]))        # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        argoman = argoman[~argoman_opened]
        ax.imshow(frame4, cmap='gray')
        # plt.show(block=True)

        argoman2 = np.copy(argoman)
        argoman3 = np.copy(argoman)

        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        pca.fit(argoman)
        arrow = pca.transform(argoman)
        arg_tip_arrow = argoman[np.argmin(arrow), :]
        arrow_tip_locations.append(arg_tip_arrow)
        ax.scatter(arg_tip_arrow[1], arg_tip_arrow[0], marker="*", s=100, c='lime')

        pca2 = PCA(n_components=2)
        pca2.fit(argoman2)
        arrow2 = pca2.transform(argoman2)
        arg_right_arrow = argoman2[np.argmin(arrow2[:,1]), :]
        # ax.scatter(arg_right_arrow[1], arg_right_arrow[0], marker="o", s=50, c='y')
        arg_left_arrow = argoman2[np.argmax(arrow2[:,1]), :]
        # ax.scatter(arg_left_arrow[1], arg_left_arrow[0], marker="o", s=50, c='y')
        # plt.savefig("/home/mahdi/IAPR/iapr-2020/tmp/before_post_process/tip_localization_frame_{}.png".format(k))
        # plt.show(block=True)

        if k==1:
            from numpy import linalg as LA
            est_arrow_plate_width = LA.norm(arg_right_arrow-arg_left_arrow)


        # Write some Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, _frame.shape[0]-25)
        fontScale = 0.45

        fontColor = (255, 255, 255)
        lineType = 1
        cv2.putText(_frame, 'frame # = {}; tip arrow location = {}; estimated arrow plate width = {:d} [pxls]'.format(k, arg_tip_arrow, int(np.rint(est_arrow_plate_width))),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        # draw linear trajectory
        if k > 1:
            # start_point = tuple(np.flip(arrow_tip_locations[-1]))
            # end_point = tuple(np.flip(arrow_tip_locations[-2]))
            # Black color in BGR
            color = (0, 100, 255)
            # Line thickness of 5 px
            thickness = 5
            for kk in range (k, 1, -1):
                _frame = cv2.line(_frame, tuple(np.flip(arrow_tip_locations[-kk+1])), tuple(np.flip(arrow_tip_locations[-kk])), color, thickness)


        cv2.imshow('frame', _frame)
        # cv2.imwrite("/home/mahdi/IAPR/iapr-2020/tmp/before_post_process/trajectory/tip_localization_frame_{}.png".format(k), frame)
        cv2.waitKey(0)
    print('mean avg_brightness = ', np.asarray(avg_brightness).mean())
    return arrow_tip_locations



def tip_tracking_3(_frame):
    frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB)

    def illumination_invariance(im):
        hsv_im = rgb2hsv(im)
        temp = np.ones(hsv_im[:, :, 2].shape)
        # 0.5958 is the mean of brightness for train video frames
        hsv_im[:, :, 2] = np.minimum(hsv_im[:, :, 2] + (0.5958 - np.mean(hsv_im[:, :, 2])), temp)
        return hsv2rgb(hsv_im)

    fig, ax = plt.subplots()
    frame = illumination_invariance(frame)

    frame2 = frame.astype('float')
    frame3 = (frame2[:, :, 0] - frame2[:, :, 1]) + (frame2[:, :, 0] - frame2[:, :, 2])
    argoman = np.transpose(
        np.asarray(np.unravel_index(np.argsort(frame3, axis=None)[::-1], frame3.shape))[:, :2000])  # pick 2000 points

    #  remove pixel outliers
    mean = np.mean(argoman, axis=0)
    standard_deviation = np.std(argoman, axis=0)
    distance_from_mean = abs(argoman - mean)
    max_deviations = 2
    not_color_outlier = np.logical_and(distance_from_mean[:, 0] < max_deviations * standard_deviation[0],
                                       distance_from_mean[:, 1] < max_deviations * standard_deviation[1])
    argoman = argoman[not_color_outlier]

    frame3 *= 0
    frame3[argoman[:, 0], argoman[:, 1]] = 1
    # used openning to remove small islands and find corresponding argoman and remove islands there as well
    from skimage.morphology import disk, opening
    frame4 = opening(frame3, disk(3))
    args_opened = np.argwhere(frame4 != frame3)
    np.in1d(argoman[:, 0], args_opened[:, 0])
    argoman_opened = np.logical_and(np.in1d(argoman[:, 0], args_opened[:, 0]), np.in1d(argoman[:, 1], args_opened[:,
                                                                                                      1]))  # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    argoman = argoman[~argoman_opened]
    ax.imshow(frame4, cmap='gray')

    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    pca.fit(argoman)
    arrow = pca.transform(argoman)
    arg_tip_arrow = argoman[np.argmin(arrow), :]

    return arg_tip_arrow







if __name__ == '__main__':
    plt.close('all')
    arrow_tip_locations = tip_tracking_2()