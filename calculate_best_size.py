
import numpy as np

def forward():

    # start_size = np.array([200.0,200.0,116.0])  # 212.0+2*86.0
    # num_layers = 3

    # start_size = np.array([300,300,116])  # 212.0+2*86.0
    # num_layers = 3

    # start_size = np.array([212+2*92,212+2*92])  # 212.0+2*86.0
    # num_layers = 4

    start_size = np.array([172.0+2*44,172.0+2*44,116.0])  # 212.0+2*86.0
    num_layers = 3

    print(start_size)

    layer_size = start_size
    for ll in range(num_layers):
        layer_size = (layer_size - 4)/2

    middle_size = layer_size -4
    print(middle_size)

    layer_size = middle_size

    for ll in range(num_layers):
        layer_size = layer_size* 2 - 4

    print(layer_size)

    print('padding')
    print(start_size-layer_size)
    print('half padding')
    print((start_size-layer_size)/2)


def backward():

    # output_size = np.array([212.0, 212.0, 28.0])  # 212.0+2*86.0
    # num_layers = 3

    output_size = np.array([100.0, 100.0, 28.0])  # 212.0+2*86.0
    num_layers = 3

    print(output_size)

    layer_size = output_size
    for ll in range(num_layers):
        layer_size = (layer_size + 4) / 2
        print(layer_size)

    middle_size = layer_size + 4
    print('middle')
    print(middle_size)

    layer_size = middle_size

    for ll in range(num_layers):
        layer_size = layer_size * 2 + 4
        print(layer_size)

    print('Beginnig')
    print(layer_size)

    print('padding')
    print(layer_size - output_size)
    print('half padding')
    print((layer_size - output_size) / 2)

if __name__ == '__main__':

    # print('forward:')
    # forward()
    print('backward:')
    backward()