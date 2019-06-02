import numpy as np
from scipy.spatial.distance import cdist
import sys

def read_inputfile(filename):
    """
    Doc file input
    :param filename: Ten file
    :return: list of headers, array data (N x M): N la so dong du lieu, M la so thuoc tinh
    """
    input_file = open(filename, "r")
    header = input_file.readline()
    header_list = header.strip(' \n').split(',')
    data_list = []
    for line in input_file:
        data_line = line.strip(' \n').split(',')
        data_line = list(map(int, data_line))
        data_list.append(data_line)
    input_file.close()
    return header_list, np.array(data_list)


def k_means(k, data):
    """
    Thuat toan k-means
    :param k: So luong cluster
    :param data: array du lieu co shape (N, M)
    :return:
    """
    # center_init la array co shape (k, M)
    center_init = data[np.random.choice(data.shape[0], k, replace=False)]
    print("Tam khoi tao:\n", center_init, "\n")
    iter = 0
    while True:
        print("Vong lap ", iter)
        iter += 1
        # Tinh khoang cach tu mot diem du lieu den tung center, co shape (N, k) voi
        # moi phan tu la khoang cach
        print("Dang tinh khoang cach...")
        distance_matrix = cdist(data, center_init)

        # Array chua nhan cluster, co shape (N, k)
        label = np.zeros((data.shape[0], k))
        print("Dang tinh nhan cua tung diem...")
        label[np.arange(0, data.shape[0]), distance_matrix.argmin(1)] = 1

        # Array chua cac tam moi, co shape (k, M)
        new_center = np.zeros((k, data.shape[1]))

        # Tinh cac center moi
        print("Dang tinh tam moi...")
        for i in range(k):
            new_center[i] = np.mean(data[label[:, i] == 1], axis=0)
        sse = np.sum(distance_matrix.min(axis=1) ** 2)
        print("Do loi SSE ", np.round(sse, 4))
        if (set(tuple(a) for a in new_center)) == (set(tuple(a) for a in center_init)):
            break
        else:
            center_init = np.copy(new_center)
    label_data = np.hstack((data, label.argmax(axis=1).reshape((data.shape[0], -1))))
    return center_init, sse, label_data


def format_output_model(model_file, header, center, error, label_data):
    file = open(model_file, "w")
    file.write(("Within cluster sum of squared errors: " + str(error) + "\n"))
    file.write("Cluster centroids:" + "\n")
    file.write("\t\t\t Cluster#" + "\n")
    k = center.shape[0]
    file.write("{0:12}\t".format("Attribute") + " ")
    for i in range(k):
        file.write("{0:8}".format(i) + " ")
    file.write("\n{0:12}\t".format(" ") + " ")
    for i in range(k):
        amount = np.sum(label_data[:, -1] == i)
        file.write("{0:8}".format(amount) + " ")
    file.write("\n")
    file.write("".center(12 + 4 + k * 8 + k, "="))
    file.write("\n")
    for i, name in enumerate(header):
        file.write("{0:12}\t".format(name) + " ")
        for k in center[:, i]:
            file.write("{0:8}".format(round(k, 4)) + " ")
        file.write("\n")
    file.close()


def format_output_file(outputfile, header, label_data):
    file = open(outputfile, "w")
    final_header = ','.join(map(str, header)) + ',Cluster' + "\n"
    file.write(final_header)
    N = label_data.shape[0]
    for i in range(0, N):
        if i != N - 1:
            data = ','.join(map(str, label_data[i])) + "\n"
        else:
            data = ','.join(map(str, label_data[i]))
        file.write(data)
    file.close()


def main(argv):
    if len(argv) != 4:
        print("Loi tham so!")
        return
    header, data = read_inputfile(argv[0])
    center, error, label_data = k_means(int(argv[3]), data)
    format_output_model(argv[1], header, center, error, label_data)
    format_output_file(argv[2], header, label_data)


if __name__ == '__main__':
    main(sys.argv[1:])



