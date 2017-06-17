import numpy as np
import cv2
import tfwrapper.utils as tf_utils

def grow_rectangle(bdr_x, bdr_y, bdr_w, bdr_h, grow_size):
    return bdr_x-grow_size, bdr_y-grow_size, bdr_w+grow_size, bdr_h+grow_size


def grow_bbox(bbox, image_size = (224,288), grow_size=(10,10)):

    bbox[0] = max(0, bbox[0]-grow_size[0])
    bbox[2] = min(image_size[1], bbox[2]+grow_size[0])

    bbox[1] = max(0, bbox[1]-grow_size[1])
    bbox[3] = min(image_size[0], bbox[3]+grow_size[1])

    return bbox


def draw_rectangle(img, bbox, colour=(255,255,255), thickness=1, lineType=8):
    # cv2.rectangle(img,(bbox[0], bbox[2]), (bbox[1],bbox[3]), colour, thickness)
    cv2.rectangle(img,(bbox[0], bbox[1]), (bbox[2], bbox[3]), color=colour, thickness=thickness, lineType=lineType)


def crop_image(image, crop_range):
    return image[crop_range[0][0]:crop_range[0][1], crop_range[1][0]:crop_range[1][1], ...]


def size_from_crop_range(crop_range):
    return crop_range[0][1]-crop_range[0][0], crop_range[1][1]-crop_range[1][0]


def convert_to_uint8(image):
    image = image - image.min()
    image = 255.0*np.divide(image.astype(np.float32),image.max())
    return image.astype(np.uint8)


def resize_image(im, size, interp=cv2.INTER_LINEAR):

    im_resized = cv2.resize(im, (size[1], size[0]), interpolation=interp)  # swap sizes to account for weird OCV API
    return im_resized


def image_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def whiten_images(X):
    """
    Helper for making the images zero mean and unit standard deviation i.e. `white`
    """

    X_white = np.zeros(X.shape, dtype=np.float32)

    for ii in range(X.shape[0]):

        Xc = X[ii,:,:,:]
        mc = Xc.mean()
        sc = Xc.std()

        Xc_white = np.divide((Xc - mc), sc)

        X_white[ii,:,:,:] = Xc_white

    return X_white.astype(np.float32)


def reshape_image_to_tensor(image):
    return np.reshape(np.float32(image), (1,1,image.shape[0], image.shape[1]))


def normalised_l2(image1, image2):
    return np.divide(cv2.norm(image1, image2, normType=cv2.NORM_L2), image1.shape[0]*image1.shape[1])


def image_l2(image1, image2):
    return cv2.norm(image1, image2, normType=cv2.NORM_L2)


def draw_labelled_bbox(img, bbox, label_name, colour=(56,244,244), title_height=12, font_scale=0.38, font_thickness=1, font_face=cv2.FONT_HERSHEY_SIMPLEX, box_thickness=1):

    label_box = [bbox[0]-(box_thickness-1), bbox[1], bbox[2]+(box_thickness-1), bbox[1]-title_height]
    draw_rectangle(img, bbox, colour=colour, thickness=box_thickness)
    draw_rectangle(img, label_box, colour=colour, thickness=-1)
    cv2.putText(
        img, label_name,
        (bbox[0]+1, bbox[1]-int(title_height/5)),
        fontFace=font_face,
        fontScale=font_scale,
        color=(0,0,0),
        thickness=font_thickness
    )


def rotate_image(img, angle, interp=cv2.INTER_LINEAR):

    rows, cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(img, rotation_matrix, (cols, rows), flags=interp)


def adjust_image_range(image, min_r, max_r):

    img_o = np.float32(image.copy())

    min_i = img_o.min()
    max_i = img_o.max()

    img_o = (img_o - min_i)*((max_r - min_r)/(max_i - min_i)) + min_r

    return img_o


def normalise_image(image):
    img_o = np.float32(image.copy())
    m = np.mean(img_o)
    s = np.std(img_o)
    return np.divide((img_o - m), s)


def rescale_image(img, scale, interp=cv2.INTER_LINEAR):

    curr_size = img.shape
    new_size = (int(float(curr_size[0])*scale[0]+0.5), int(float(curr_size[1])*scale[1]+0.5))
    img_resized = cv2.resize(img, (new_size[1], new_size[0]), interpolation=interp)  # swap sizes to account for weird OCV API
    return img_resized

def rescale_labels_lisa_style(label_map, scale, num_labels):

    label_map_one_hot = tf_utils.to_onehot_image(label_map, num_labels)
    out_labels = []
    for ll in range(num_labels):
        lbl = np.squeeze(label_map_one_hot[:,:,ll])
        lbl_sc = rescale_image(lbl.astype(np.float), scale)
        out_labels.append(lbl_sc)
    out_array_one_hot = np.asarray(out_labels)
    out_array = np.argmax(out_array_one_hot, axis=0).astype(np.uint8)

    return out_array

def resize_labels_lisa_style(label_map, size, num_labels):

    label_map_one_hot = tf_utils.to_onehot_image(label_map, num_labels)
    out_labels = []
    for ll in range(num_labels):
        lbl = np.squeeze(label_map_one_hot[:,:,ll])
        lbl_sc = resize_image(lbl.astype(np.float), size)
        out_labels.append(lbl_sc)
    out_array_one_hot = np.asarray(out_labels)
    out_array = np.argmax(out_array_one_hot, axis=0).astype(np.uint8)

    return out_array