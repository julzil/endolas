import imageio
import argparse
import os
from matplotlib import pyplot as plt
import pdb

if __name__ == "__main__":
    reader_list=[]
    for i in range(1,16):
        reader_list.append(imageio.imread("{}.png".format(i)))

    writer = imageio.get_writer("out.gif",
                                format="gif",
                                fps=2)#,
                                #codec="rawvideo")

    for index, frame in enumerate(reader_list):
        print("Writing frame {}".format(index))
        writer.append_data(frame)

    writer.close()
    print("Finished")
