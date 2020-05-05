from skimage import io
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.morphology import disk, closing


def tip_tracking():

    cap = cv2.VideoCapture('/home/mahdi/IAPR/iapr-2020/data/project_data/robot_parcours_1.avi')

    k = 0
    while(cap.isOpened()):
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        ret, _frame = cap.read()
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
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(frame3, cmap='gray')

        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        pca.fit(argoman)
        arrow = pca.transform(argoman)
        arg_tip_arrow = argoman[np.argmin(arrow), :]
        # frame3[arg_tip_arrow[0], arg_tip_arrow[1]]=
        ax.scatter(arg_tip_arrow[1], arg_tip_arrow[0], marker="*", s=100, c='lime')

        plt.show()


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

        cv2.imshow('frame', _frame)













if __name__ == '__main__':
    plt.close('all')
    tip_tracking()